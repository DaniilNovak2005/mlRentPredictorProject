import asyncio
import json
import nodriver as uc
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
import sys
import os
import json
import math
import urllib.parse

global initialized

def create_zillow_search_query(location, miles=10, pageNumber = 1 ):
    """
    Converts a location object into a Zillow search query dictionary.

    Args:
        location (dict): A dictionary containing 'latitude' and 'longitude'.
        miles (int): The side length of the square bounding box in miles.

    Returns:
        dict: A dictionary formatted for a Zillow search query.
    """
    lat = location['latitude']
    lon = location['longitude']

    # Earth radius in miles
    earth_radius = 3958.8

    # Angular distance in radians on a great circle
    lat_change = miles / earth_radius
    lon_change = miles / (earth_radius * math.cos(math.radians(lat)))

    # New latitude and longitude
    north = lat + .015 #math.degrees(lat_change)
    south = lat - .015 #math.degrees(lat_change)
    east = lon + .015 #math.degrees(lon_change)
    west = lon - .015 #math.degrees(lon_change)

    reusult = {
        "isMapVisible": True,
        "mapBounds": {
            "north": north,
            "south": south,
            "east": east,
            "west": west
        },
        "filterState": {
            "mf": {"value": False},
            "land": {"value": False},
            "manu": {"value": False},
            "fr": {"value": True},
            "fsba": {"value": False},
            "fsbo": {"value": False},
            "nc": {"value": False},
            "cmsn": {"value": False},
            "auc": {"value": False},
            "fore": {"value": False}
        },
        "isListVisible": True,
        "mapZoom": 13,
    }

    if pageNumber > 1:
        reusult['pagination'] = {
            'currentPage' : pageNumber,
        }
    return reusult


BASE_URL = "https://www.zillow.com/homes/for_rent/"
OUTPUT_FILE = "output.parquet"
initialized = 0

# Create a lock to manage file access
file_lock = asyncio.Lock()
startup_lock = asyncio.Lock()

def rfind_custom(text: str, sub: str, start: int = 0, end: int = None) -> int:
    if end is None:
        end = len(text)

    search_slice = text[start:end]
    sub_len = len(sub)
    slice_len = len(search_slice)

    for i in range(slice_len - sub_len, -1, -1):
        if search_slice[i:i + sub_len] == sub:
            return i + start
            
    return -1


async def check_for_error(page):
    """Checks for various error conditions on the page."""
    try:
        # Check for "Access Denied" in the page title
        title = await page.evaluate("document.title")
        if title and "been denied" in title.lower():
            logging.warning("Access Denied detected in page title.")
            return True

        # Check for "something broke" message
        if "something broke" in await page.get_content():
            logging.info("Found 'something broke' error message.")
            return True

        # Check for main-frame-error element
        error_element = await page.select("#main-frame-error", timeout=1)
        if error_element:
            logging.warning("Found main-frame-error element.")
            return True

        return False
    except Exception as e:
        logging.error(f"An error occurred during error check: {e}")
        return False

async def check_for_captcha(page):
    """Checks for the presence of a captcha element."""
    try:
        captcha_element = await page.select("#px-captcha-wrapper", timeout=1)
        if captcha_element:
            logging.info("Captcha element found.")
            return True
        return False
    except Exception as e:
        logging.error(f"An error occurred during captcha check: {e}")
        return False
    
async def houses_selected(page):
    inName = False
    try:
        search_title = await page.select(".search-title", timeout=10)
        if search_title:
            inName = "house" in (search_title.text).lower()
    except Exception as e:
        pass

    return inName


async def set_home_type_filter(page, proxy, lock):
    """Sets the home type filter to exclude apartments/condos."""
    home_type_button = None
    home_type_section = None
    apartment_condo_checkbox = None
    try:
        await asyncio.sleep(5)
        while await check_for_captcha(page) or await check_for_error(page):
            logging.warning(f"Captcha or error detected for proxy {proxy}. Waiting 5 minutes and reloading.")
            await asyncio.sleep(300)
            await page.reload()

        home_type_button_selector = '[data-test="home-type-filters-button"]'
        try:
            home_type_button = await page.select(home_type_button_selector, timeout=10)
        except Exception:
            logging.error(f"Proxy {proxy} could not find home type filter button.")
            return False

        if home_type_button:
            logging.info(f"Proxy {proxy} clicking home type filter button.")
            await home_type_button.click()
            await asyncio.sleep(2)

            home_type_section_selector = "section#home-type"
            try:
                home_type_section = await page.select(home_type_section_selector, timeout=10)
            except Exception:
                logging.error(f"Proxy {proxy} could not find home type section.")
                return False

            if home_type_section:
                apartment_condo_selector = "#home-type_isApartmentOrCondo"
                try:
                    apartment_condo_checkbox = await page.select(apartment_condo_selector, timeout=10)
                except Exception:
                    logging.error(f"Proxy {proxy} could not find apartment/condo checkbox.")
                    return False

                if apartment_condo_checkbox:
                    attrs = apartment_condo_checkbox.attributes
                    attr_dict = {attrs[i]: attrs[i+1] for i in range(0, len(attrs), 2)}
                    if 'checked' in attr_dict:
                        logging.info(f"Proxy {proxy} clicking apartment/condo checkbox to uncheck it.")
                        await asyncio.sleep(.5)
                        await apartment_condo_checkbox.click()
                    else:
                        logging.info(f"Proxy {proxy} apartment/condo checkbox is not checked.")
                
                await home_type_button.click() # Close the filter
                await asyncio.sleep(2)
                logging.info(f"Proxy {proxy} successfully set home type filter.")
                return True
        
        return False
    except Exception as e:
        logging.error(f"An error occurred in set_home_type_filter: {e}")
        return False
    finally:
        if lock.locked():
            lock.release()
            logging.info(f"Proxy {proxy} released the lock.")

async def fetch_zillow_data(proxy, locations, index, startup_lock):
    global initialized
    """
    Uses a proxy to fetch Zillow data for a list of locations.
    """
    await startup_lock.acquire()
    logging.info(f"Worker with proxy {proxy} starting for {len(locations)} locations.")
    browser = None
    scraped_links = set()
    try:
        logging.info(f"Proxy {proxy} acquiring lock to  browser.")
        userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        logging.info(f"Proxy {proxy} started browser, releasing lock.")
        EXT_PATH = r'/root/.config/google-chrome/Default/Extensions/olnbjpaejebpnokblkepbphhembdicik/0.1.8_0'
        browser_counter = 0
        while browser is None:
            browser_counter += 1
            logging.info(f"Attempting to start browser (attempt {browser_counter}) with proxy {proxy}...")
            try:
                browser = await uc.start(browser_args=[f"--proxy-server=http://{proxy}","--disable-dev-shm-usage",f"--load-extension{EXT_PATH}"])
                logging.info(f"Browser started successfully with proxy {proxy}.")
            except Exception as e:
                logging.error(f"Failed to start browser with proxy {proxy}: {e}")
                await asyncio.sleep(30)
        filter_set = False
        i = 0
        location = None

        while i < len(locations):
            location = locations[i]
            
            logging.info(f"Scraping {location['city']}...")
            if i > 0:
                await asyncio.sleep(5)

            query_dict = create_zillow_search_query(location, 10, 1)
            json_query = json.dumps(query_dict, separators=(',', ':'))
            url_encoded_query = urllib.parse.quote(json_query)
            full_url = f"{BASE_URL}?searchQueryState={url_encoded_query}"
            
            logging.info(f"Proxy {proxy} fetching initial URL: {full_url}")
            page = await browser.get(full_url)
            await asyncio.sleep(5)
            
            if not page:
                logging.warning(f"Proxy {proxy} failed to load initial page for {location['city']}")
                i += 1
                continue

            if not filter_set:
                await page.set_window_size(0, 0, 1920, 1080)
                initialized += 1
                await page.activate()
                await asyncio.sleep(5)
                await set_home_type_filter(page, proxy, startup_lock)
                filter_set = True
            page_num = 1
            while True:
                try:
                    await asyncio.sleep(2)
                    if page_num > 1:
                        currentURL = page.target.url
                        currentNoQueryURL = currentURL.split('?')[0]

                        # If the URL for the previous page has a page number, strip it.
                        if "_p" in currentNoQueryURL:
                            p_index = currentNoQueryURL.rfind("_p")
                            base_url_end = currentNoQueryURL.rfind('/', 0, p_index)
                            if base_url_end != -1:
                                currentNoQueryURL = currentNoQueryURL[:base_url_end + 1]
                        
                        if not currentNoQueryURL.endswith('/'):
                            currentNoQueryURL += '/'

                        query_dict = create_zillow_search_query(location, 10, page_num)
                        print(query_dict['pagination'])
                        json_query = json.dumps(query_dict, separators=(',', ':'))
                        url_encoded_query = urllib.parse.quote(json_query)
                        new_url = f"{currentNoQueryURL}{page_num}_p/?searchQueryState={url_encoded_query}"
                            
                        page = await browser.get(new_url)
                    captcha_found = False
                    duplicate_found = False

                    if await check_for_captcha(page):
                        logging.warning(f"Captcha detected for proxy {proxy}. Waiting 5 minutes and reloading.")
                        await asyncio.sleep(60)
                        await page.reload()
                        continue

                    title = await page.evaluate("document.title")
                    logging.info(f"Proxy {proxy} got title: {title} for page {page_num}")
                    
                    await asyncio.sleep(0.1)
                    if "something broke" in await page.get_content():
                        logging.info(f"Proxy {proxy} ran into error 404, quitting.")
                        break

                    # Check for errorShortDesc element
                    while True:
                        try:
                            error_element = await page.select("#main-frame-error", timeout=2)
                            if error_element:
                                logging.warning(f"Proxy {proxy} found errorShortDesc element. Waiting 10 seconds and reloading.")
                                await asyncio.sleep(10)
                                await page.reload()
                            else:
                                break  # Element not found, exit the loop
                        except Exception as e:
                            # An exception likely means the element is not present
                            break
                            
                    if page_num == 1:
                        # Click the listing type filter button
                        filter_button_selector = '[data-test="listing-type-filters-button"]'
                        filter_button = await page.select(filter_button_selector, timeout=10)

                        if filter_button:
                            button_text = filter_button.text
                            if "rent" not in button_text.lower():
                                while True:
                                    if await check_for_captcha(page):
                                        captcha_found = True
                                        break
                                    try:
                                        logging.info(f"Proxy {proxy} clicking filter button.")
                                        await filter_button.click()
                                        break
                                    except Exception as e:
                                        logging.warning(f"Proxy {proxy} could not find filter button, retrying... Error: {e}")
                                        await asyncio.sleep(0.1)
                                
                                    if captcha_found:
                                        logging.warning(f"Captcha detected for proxy {proxy}. Waiting 5 minutes and restarting.")
                                        await asyncio.sleep(60)
                                        await page.reload()
                                        continue

                                # Wait for and click the 'For Rent' radio button
                                for_rent_selector = "#isForRent"
                                while True:
                                    if await check_for_captcha(page):
                                        captcha_found = True
                                        break
                                    for_rent_button = await page.select(for_rent_selector, timeout=2)
                                    if for_rent_button:
                                        logging.info(f"Proxy {proxy} clicking 'For Rent' button.")
                                        await for_rent_button.click()
                                        break
                                    else:
                                        logging.warning(f"Proxy {proxy} could not find 'For Rent' button, re-clicking filter.")
                                        await filter_button.click()
                                        await asyncio.sleep(0.1)

                                    if captcha_found:
                                        logging.warning(f"Proxy {proxy}. Waiting 5 minutes and restarting.")
                                        await asyncio.sleep(60)
                                        await page.reload()
                                        continue

                                # Click the filter button again to close
                                await filter_button.click()
                        

                    logging.info(f"Proxy {proxy} waiting for 10 seconds before scrolling.")
                    await asyncio.sleep(10)

                    logging.info(f"Proxy {proxy} scrolling down.")
                    for j in range(0, 10):
                        await page.evaluate(f"window.scrollBy(0, {j * 100});")
                        await asyncio.sleep(0.1)

                    # Select the list and process its children
                    await asyncio.sleep(2)
                    list_selector = "ul.List-c11n-8-109-3__sc-1smrmqp-0.StyledSearchListWrapper-srp-8-109-3__sc-14brvob-0.ffjqkg.kyEqXS.photo-cards"
                    property_list = await page.select(list_selector, timeout=20)
                    links_found_on_page = 0
                    if property_list:
                        apartment_found = False
                        children = property_list.children
                        for child in children:
                            html = await child.get_html()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            ld_script = soup.find('script', {'type': 'application/ld+json'})
                            if ld_script and ld_script.string:
                                try:
                                    ld_json_data = json.loads(ld_script.string)
                                    url = ld_json_data.get('url', 'N/A')
                                    ld_type = ld_json_data.get('@type', 'N/A')
                                    address = ld_json_data.get('address', {})
                                    geo = ld_json_data.get('geo', {})

                                    if url not in scraped_links:
                                        scraped_links.add(url)
                                        links_found_on_page += 1
                                        logging.info(f"Proxy {proxy} found Address: {address.get('streetAddress', 'N/A')}, Link: {url}")

                                    new_data = {
                                        'City': [location['city']],
                                        'URL': [url],
                                        'Type': [ld_type],
                                        'Address': [json.dumps(address)],
                                        'Geo': [json.dumps(geo)]
                                    }
                                    new_df = pd.DataFrame(new_data)

                                    async with file_lock:
                                        try:
                                            if os.path.exists(OUTPUT_FILE):
                                                table = pq.read_table(OUTPUT_FILE)
                                                existing_df = table.to_pandas()
                                                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                                                pq.write_table(pa.Table.from_pandas(combined_df), OUTPUT_FILE)
                                            else:
                                                pq.write_table(pa.Table.from_pandas(new_df), OUTPUT_FILE)
                                        except Exception as e:
                                            logging.error(f"Error writing to parquet file: {e}")
                                except json.JSONDecodeError:
                                    logging.error(f"Proxy {proxy} failed to parse ld+json")
                        
                        if duplicate_found:
                            break
                    else:
                        logging.warning(f"Proxy {proxy} could not find the property list.")

                    if links_found_on_page == 0:
                        logging.info(f"No new links found for {location['city']} on page {page_num}. Moving to next location.")
                        break
                    
                    page_num += 1

                except uc.core.connection.ProtocolException as e:
                    logging.error(f"ProtocolException caught: {e}. Reloading page and continuing.")
                    await asyncio.sleep(1)
                    try:
                        await page.reload()
                    except Exception as reload_e:
                        logging.error(f"Failed to reload page after ProtocolException: {reload_e}")
                        # If reload fails, break the inner loop to try the next location
                        break
                    continue
                except Exception as e:
                    logging.error(f"An unexpected error occurred in the main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            i += 1


    except StopIteration:
        logging.warning(f"Proxy {proxy} encountered a StopIteration, likely a page load issue.")
    except Exception as e:
        import traceback
        if location:
            logging.error(f"An error occurred with proxy {proxy} for location {location.get('city', 'N/A')}: {e}")
        else:
            logging.error(f"An error occurred with proxy {proxy} before location was set: {e}")
        traceback.print_exc()
    finally:
        if browser:
            browser.stop()
    logging.info(f"Worker with proxy {proxy} finished.")


async def main():
    # Set up logging
    log_filename = f"{sys.argv[0].split('.')[0]}.log"
    # Clear the log file
    with open(log_filename, 'w'):
        pass
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # --- Caching Implementation ---
    processed_cities = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            table = pq.read_table(OUTPUT_FILE, columns=['City'])
            df = table.to_pandas()
            if not df.empty:
                processed_cities = set(df['City'].unique())
        except Exception as e:
            logging.error(f"Could not read parquet file for caching: {e}")

    with open('jsonData/config.json', 'r') as f:
        config = json.load(f)
    proxies = config['proxies']

    with open('jsonData/locations.json', 'r') as f:
        locations_data = json.load(f)['locations']

    # Filter out already processed cities
    locations_to_scrape = [loc for loc in locations_data if loc['city'] not in processed_cities]
    logging.info(f"Found {len(locations_to_scrape)} new cities to scrape.")

    if not locations_to_scrape:
        logging.info("No new cities to scrape. Exiting.")
        return

    num_proxies = len(proxies)
    if num_proxies == 0:
        logging.error("No proxies found in config.json")
        return
        
    locations_per_proxy = -(-len(locations_to_scrape) // num_proxies)  # Ceiling division
    
    startup_lock = asyncio.Lock()
    tasks = []
    for i, proxy in enumerate(proxies):
        while startup_lock.locked():
            await asyncio.sleep(1)

        start_index = i * locations_per_proxy
        end_index = start_index + locations_per_proxy
        
        locations_chunk = locations_to_scrape[start_index:end_index]
        if locations_chunk:
            task = asyncio.create_task(fetch_zillow_data(proxy, locations_chunk, i, startup_lock))
            tasks.append(task)
        
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())