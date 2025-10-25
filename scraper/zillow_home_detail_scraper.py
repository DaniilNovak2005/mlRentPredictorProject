import asyncio
import json
import zendriver as uc
import urllib.parse
from bs4 import BeautifulSoup
import csv
import logging
import sys
from datetime import datetime
import statistics
import requests
import time
import os

BASE_URL = "https://www.zillow.com"
INPUT_FILE = "csvData/inputToAnalyzeV1.csv"
OUTPUT_FILE = "csvData/floridaHomeDetailsV3.csv"
WAIT_TIME = 5

# Create a lock to manage file access
file_lock = asyncio.Lock()

def logLink(proxy, link):
    try:
        logContent = {
            "content": f"{proxy} Link Scraped{link}",
        }
        requests.post("DISCORD_PROGRESS",data=logContent)
    except Exception as e:
        pass

async def check_for_captcha(page):
    """Checks for captcha, error codes, or access denied in title."""
    logging.info("Checking for captcha, error, or access denied...")
    try:
        # Check page title for access denied
        title = await page.evaluate("document.title")
        if title and "been denied" in title.lower():
            logging.warning(f"Access denied detected in page title: '{title}'")
            return "captcha"
        
        # Check for captcha elements
        if await page.select("div#px-captcha-wrapper", timeout=1.5):
            logging.info("Captcha element found.")
            return "captcha"
        if await page.select("div#error-code", timeout=.5):
            logging.info("Error code div found.")
            return "error"


        logging.info("No captcha, error, or access denied found.")
        return None
    except Exception as e:
        logging.error(f"An error occurred during captcha/error check: {e}")
        return None

def calculate_price_history_stats(price_history):
    """
    Calculates rental statistics from Zillow's price history.
    """
    if not price_history:
        return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', '[]', 'N/A'

    try:
        sorted_history = sorted(price_history, key=lambda x: x.get('time', 0))
        
        events = []
        for event in sorted_history:
            event_data = {
                'type': event.get('event'),
                'date': event.get('time')
            }
            if event.get('price'):
                event_data['price'] = event.get('price')
            if event.get('priceChangeRate'):
                event_data['priceChangeRate'] = event.get('priceChangeRate')
            events.append(event_data)
        
        rental_periods = []
        i = 0
        while i < len(sorted_history):
            event = sorted_history[i]
            if event.get('event') == 'Listed for rent':
                start_time = event.get('time')
                start_price = event.get('price')
                price_changes = []
                
                j = i + 1
                while j < len(sorted_history):
                    end_event = sorted_history[j]
                    if end_event.get('event') == 'Price change':
                        price_changes.append({
                            'time': end_event.get('time'),
                            'price': end_event.get('price')
                        })
                    elif end_event.get('event') == 'Listing removed':
                        end_time = end_event.get('time')
                        if start_time and end_time:
                            rental_periods.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'start_price': start_price,
                                'price_changes': price_changes,
                                'end_price': price_changes[-1]['price'] if price_changes else start_price
                            })
                        i = j
                        break
                    j += 1
            i += 1

        if not rental_periods:
            return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', json.dumps(events), 'N/A'

        # Combine rentals re-rented in less than 90 days
        merged_periods = []
        if rental_periods:
            rental_periods.sort(key=lambda x: x['start_time'])
            
            current_period = rental_periods[0]
            
            for i in range(1, len(rental_periods)):
                next_period = rental_periods[i]
                time_diff_days = (datetime.fromtimestamp(next_period['start_time'] / 1000) - datetime.fromtimestamp(current_period['end_time'] / 1000)).days
                
                if time_diff_days < 90:
                    # Merge periods
                    current_period['end_time'] = next_period['end_time']
                    current_period['price_changes'].extend(next_period['price_changes'])
                    current_period['end_price'] = next_period['end_price']
                else:
                    merged_periods.append(current_period)
                    current_period = next_period
            merged_periods.append(current_period)

        if not merged_periods:
            return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', json.dumps(events), 'N/A'

        # Calculate stats from merged periods
        durations = [(datetime.fromtimestamp(p['end_time'] / 1000) - datetime.fromtimestamp(p['start_time'] / 1000)).days for p in merged_periods]
        
        most_recent_period = merged_periods[-1]
        most_recent_time_on_market = durations[-1] if durations else 'N/A'
        
        previous_days_on_market = 'N/A'
        if len(durations) > 1:
            previous_days_on_market = durations[-2]

        average_time_on_market = round(statistics.mean(durations), 2) if durations else 'N/A'
        
        price_lowered_count = 0
        price_decreases = []
        last_price = most_recent_period['start_price']
        for change in most_recent_period['price_changes']:
            if change.get('price') and last_price and change.get('price') < last_price:
                price_lowered_count += 1
                price_decreases.append(last_price - change['price'])
            last_price = change.get('price')
            
        average_price_decrease = round(statistics.mean(price_decreases), 2) if price_decreases else 0

        last_rental_price = most_recent_period['end_price']
        times_rented = len(merged_periods)
        
        last_rental_time = most_recent_period['end_time']
        if last_rental_time:
            last_rental_date = datetime.fromtimestamp(last_rental_time / 1000)
            years_since_rented = round((datetime.now() - last_rental_date).days / 365.25, 2)
        else:
            years_since_rented = 'N/A'
            
        return (last_rental_price, times_rented, years_since_rented,
                most_recent_time_on_market, average_time_on_market,
                price_lowered_count, average_price_decrease, json.dumps(events), previous_days_on_market)

    except (KeyError, TypeError, IndexError):
        return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', '[]', 'N/A'

def calculate_tax_history_stats(tax_history):
    """
    Calculates financial statistics from Zillow's tax history.
    """
    if not tax_history or len(tax_history) < 2:
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

    try:
        sorted_history = sorted(tax_history, key=lambda x: x['time'])
    except KeyError:
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

    # CAGR Calculation
    ending_value = sorted_history[-1].get('value')
    beginning_value = sorted_history[0].get('value')
    
    if not ending_value or not beginning_value or not sorted_history[-1].get('time') or not sorted_history[0].get('time'):
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

    num_years = (datetime.fromtimestamp(sorted_history[-1]['time'] / 1000) - datetime.fromtimestamp(sorted_history[0]['time'] / 1000)).days / 365.25
    
    if beginning_value == 0 or num_years == 0:
        cagr = 'N/A'
    else:
        cagr = round(((ending_value / beginning_value) ** (1 / num_years)) - 1, 4)

    # Annual Increases and Standard Deviations
    value_increases = [h['valueIncreaseRate'] for h in sorted_history if 'valueIncreaseRate' in h]
    tax_increases = [h['taxIncreaseRate'] for h in sorted_history if 'taxIncreaseRate' in h]

    avg_value_increase = round(statistics.mean(value_increases), 4) if value_increases else 'N/A'
    avg_tax_increase = round(statistics.mean(tax_increases), 4) if tax_increases else 'N/A'
    
    std_dev_value_increase = round(statistics.stdev(value_increases), 4) if len(value_increases) > 1 else 0.0
    std_dev_tax_increase = round(statistics.stdev(tax_increases), 4) if len(tax_increases) > 1 else 0.0

    return cagr, avg_value_increase, avg_tax_increase, std_dev_value_increase, std_dev_tax_increase

async def fetch_home_details(proxy, urls, start_time):
    """
    Uses a proxy to fetch Zillow home details for a list of URLs.
    """
    process_start_datetime = datetime.now()
    logging.info(f"Worker with proxy {proxy} starting for {len(urls)} URLs at {process_start_datetime}.")
    browser = None
    try:
        browser = await uc.start(browser_args=[f"--proxy-server=http://{proxy}","--disable-application-cache","--disable-features=TabDiscarding"],use_cache=False)
        i = 0
        while i < len(urls):
            # Check if 15 minutes have elapsed
            if time.time() - start_time >= 900:  # 900 seconds = 15 minutes
                logging.info(f"15-minute limit reached for proxy {proxy}. Breaking loop.")
                break
            url = urls[i]
            if not url.startswith("http"):
                full_url = BASE_URL + url
            else:
                full_url = url

            logging.info(f"Proxy {proxy} fetching URL: {full_url}")
            
            try:
                page = await browser.get(full_url)
                if not page:
                    logging.warning(f"Proxy {proxy} failed to load page for {full_url}")
                    i += 1
                    await browser.sleep(WAIT_TIME)
                    continue

                try:
                    status = await check_for_captcha(page)
                    if status == "captcha":
                        logging.warning(f"Captcha detected for proxy {proxy}. Waiting 5 minutes and reloading.")
                        await browser.sleep(300)
                        await page.reload()
                        continue
                    elif status == "error":
                        logging.warning(f"Page error detected for proxy {proxy}. Waiting 5 seconds and reloading.")
                        await browser.sleep(WAIT_TIME)
                        await page.reload()
                        continue
                except Exception as e:
                    logging.error(f"Error occurred in check_for_captcha for {full_url}: {e}. Continuing...")
                    continue

                
                # Wait for the script element to be loaded, then get content
                tries = 0
                nextdata = await page.select("#__NEXT_DATA__", timeout=300)
                
                if nextdata:
                    if tries >= 5:
                        break
                    tries += 1
                    await asyncio.sleep(.5)
                    page_content = await page.get_content()
                    soup = BeautifulSoup(page_content, 'html.parser')
                    script_element = soup.find('script', {'id': '__NEXT_DATA__'})
                    print(script_element)
                    
                    if script_element and script_element.string:
                        script_text = script_element.string
                        try:
                            json_data = json.loads(script_text)
                        except json.JSONDecodeError:
                            logging.error(f"JSON parsing error for {full_url}. Reloading and retrying after 5 seconds.")
                            await asyncio.sleep(WAIT_TIME)
                            await page.reload()
                            continue
                    else:
                        logging.warning(f"Could not extract text from '__NEXT_DATA__' script for {full_url}")
                        json_data = None
                else:
                    logging.warning(f"Could not find '__NEXT_DATA__' script for {full_url}. Waiting 1 minute then continuing.")
                    json_data = None
                    await asyncio.sleep(60)

                if json_data:
                    # Check if 15 minutes have elapsed before starting data extraction
                    if time.time() - start_time >= 900:  # 900 seconds = 15 minutes
                        logging.info(f"15-minute limit reached during data extraction for proxy {proxy}. Skipping URL.")
                        break

                    for attempt in range(3):
                        try:
                            property_data_str = json_data['props']['pageProps']['componentProps']['gdpClientCache']
                            property_data = json.loads(property_data_str)
                            zpid_key = list(property_data.keys())[0]
                            property_info = property_data[zpid_key]
                            details = property_info['property']
                            address = details.get('address', {})
                            street_address = address.get('streetAddress', 'N/A')
                            city = address.get('city', 'N/A')
                            state = address.get('state', 'N/A')
                            zipcode = address.get('zipcode', 'N/A')
                            
                            price = details.get('price', 'N/A')
                            bedrooms = details.get('bedrooms', 'N/A')
                            bathrooms = details.get('bathrooms', 'N/A')
                            living_area = details.get('livingAreaValue', 'N/A')
                            home_type = details.get('homeType', 'N/A')
                            home_status = details.get('homeStatus', 'N/A')
                            year_built = details.get('yearBuilt', 'N/A')
                            property_tax_rate = details.get('propertyTaxRate', 'N/A')
                            latitude = details.get('latitude', 'N/A')
                            longitude = details.get('longitude', 'N/A')
                            days_on_zillow = details.get('daysOnZillow', 'N/A')
                            zestimate = details.get('zestimate', 'N/A')
                            rent_zestimate = details.get('rentZestimate', 'N/A')
                            tax_assessed_value = details.get('taxAssessedValue', 'N/A')
                            lot_area_value = details.get('lotAreaValue', 'N/A')
                            date_details_fetched = datetime.now().strftime("%Y-%m-%d")

                            image_urls = []
                            if 'responsivePhotos' in details and isinstance(details['responsivePhotos'], list):
                                for photo in details['responsivePhotos']:
                                    if isinstance(photo, dict) and 'mixedSources' in photo:
                                        if 'jpeg' in photo['mixedSources'] and photo['mixedSources']['jpeg']:
                                            image_urls.append(photo['mixedSources']['jpeg'][-1]['url'])
                            
                            if not image_urls and 'originalPhotos' in details and isinstance(details['originalPhotos'], list):
                                for photo in details['originalPhotos']:
                                     if isinstance(photo, dict) and 'mixedSources' in photo:
                                        if 'jpeg' in photo['mixedSources'] and photo['mixedSources']['jpeg']:
                                            image_urls.append(photo['mixedSources']['jpeg'][-1]['url'])
                            image_urls_json = json.dumps(image_urls)
                            
                            # Get property tax from tax history if available
                            tax_history = details.get('taxHistory', [])
                            property_tax = 'N/A'
                            if tax_history and len(tax_history) > 0:
                                property_tax = tax_history[0].get('taxPaid', 'N/A')
                            
                            price_history = details.get('priceHistory', [])
                            (last_rental_price, times_rented, years_since_rented,
                            most_recent_time_on_market, average_time_on_market,
                            price_lowered_count, average_price_decrease, events_json, previous_days_on_market) = calculate_price_history_stats(price_history)

                            if home_status != 'FOR_RENT':
                                price = last_rental_price
                                previous_days_on_market = most_recent_time_on_market
                                most_recent_time_on_market = days_on_zillow
                                
                            cagr, avg_value_increase, avg_tax_increase, std_dev_value_increase, std_dev_tax_increase = calculate_tax_history_stats(tax_history)

                            # New fields
                            walk_score_tag = soup.find('a', {'aria-describedby': 'walk-score-text'})
                            walk_score = walk_score_tag.text if walk_score_tag else 'N/A'
                            transit_score_tag = soup.find('a', {'aria-describedby': 'transit-score-text'})
                            transit_score = transit_score_tag.text if transit_score_tag else 'N/A'
                            bike_score_tag = soup.find('a', {'aria-describedby': 'bike-score-text'})
                            bike_score = bike_score_tag.text if bike_score_tag else 'N/A'

                            schools_data = details.get('schools', [])
                            school_info = {'Primary': {}, 'Middle': {}, 'High': {}}
                            for s in schools_data:
                                level = s.get('level', 'N/A')
                                if level in school_info and not school_info[level]:
                                    school_info[level] = {
                                        'distance': s.get('distance', 'N/A'),
                                        'rating': s.get('rating', 'N/A')
                                    }
                            
                            primary_school_distance = school_info['Primary'].get('distance', 'N/A')
                            primary_school_rating = school_info['Primary'].get('rating', 'N/A')
                            middle_school_distance = school_info['Middle'].get('distance', 'N/A')
                            middle_school_rating = school_info['Middle'].get('rating', 'N/A')
                            high_school_distance = school_info['High'].get('distance', 'N/A')
                            high_school_rating = school_info['High'].get('rating', 'N/A')

                            reso_facts = details.get('resoFacts', {})
                            appliances = ", ".join(reso_facts.get('appliances', [])) if reso_facts.get('appliances') else 'N/A'
                            cooling = ", ".join(reso_facts.get('cooling', [])) if reso_facts.get('cooling') else 'N/A'
                            heating = ", ".join(reso_facts.get('heating', [])) if reso_facts.get('heating') else 'N/A'
                            laundry = ", ".join(reso_facts.get('laundryFeatures', [])) if reso_facts.get('laundryFeatures') else 'N/A'
                            parking = ", ".join(reso_facts.get('parkingFeatures', [])) if reso_facts.get('parkingFeatures') else 'N/A'


                            async with file_lock:
                                with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([
                                        street_address, city, state, zipcode, price, bedrooms, bathrooms,
                                        living_area, home_type, year_built, property_tax, property_tax_rate,
                                        latitude, longitude, days_on_zillow, last_rental_price, times_rented, years_since_rented,
                                        cagr, avg_value_increase, avg_tax_increase, std_dev_value_increase, std_dev_tax_increase,
                                        walk_score, transit_score, bike_score,
                                        primary_school_distance, primary_school_rating,
                                        middle_school_distance, middle_school_rating,
                                        high_school_distance, high_school_rating,
                                        appliances, cooling, heating, laundry, parking,
                                        zestimate, rent_zestimate, tax_assessed_value, lot_area_value,
                                        most_recent_time_on_market, average_time_on_market,
                                        price_lowered_count, average_price_decrease,
                                        date_details_fetched, image_urls_json, events_json, previous_days_on_market,
                                        full_url
                                    ])
                            logging.info(f"Successfully scraped: {street_address}")
                            logLink(proxy,url)
                            break  # Success, exit retry loop
                        except (KeyError, IndexError, TypeError) as e:
                            logging.error(f"Could not extract data for {full_url} on attempt {attempt + 1}. Error: {e}")
                            if attempt < 2:
                                logging.info("Retrying after 5 seconds...")
                                await browser.sleep(WAIT_TIME)
                            else:
                                logging.error(f"Failed to extract data for {full_url} after 3 attempts.")

            except Exception as e:
                logging.error(f"An error occurred while processing {full_url}: {e}")

            i += 1
            logging.info("Waiting for 5 seconds before next request...")
            await browser.sleep(WAIT_TIME)

    except Exception as e:
        import traceback
        logging.error(f"An error occurred with proxy {proxy}: {e}")
        traceback.print_exc()
    finally:
        if browser:
            await browser.stop()
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

    while True:  # Continuous loop
        start_time = time.time()
        logging.info(f"Starting new scraping cycle at {datetime.now()}")

        # --- Caching Implementation ---
        scraped_urls = set()
        try:
            with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    next(reader)  # Skip header
                except StopIteration:
                    pass  # Empty file
                for row in reader:
                    if row:
                        scraped_urls.add(row[-1])  # URL is the last column
        except FileNotFoundError:
            # If the file doesn't exist, create it and write the header
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Street Address', 'City', 'State', 'Zipcode', 'Price', 'Bedrooms', 'Bathrooms',
                    'Living Area', 'Home Type', 'Year Built', 'Property Tax', 'Property Tax Rate',
                    'Latitude', 'Longitude', 'Days on Zillow', 'Last Rental Price', 'Times Rented',
                    'Years Since Rented', 'CAGR', 'Avg Annual Value Increase',
                    'Avg Annual Tax Increase', 'Std Dev Value Increase', 'Std Dev Tax Increase',
                    'Walk Score', 'Transit Score', 'Bike Score',
                    'Primary School Distance', 'Primary School Rating',
                    'Middle School Distance', 'Middle School Rating',
                    'High School Distance', 'High School Rating',
                    'Appliances', 'Cooling', 'Heating', 'Laundry', 'Parking',
                    'Zestimate', 'Rent Zestimate', 'Tax Assessed Value', 'Lot Area Value',
                    'Most Recent Time on Market', 'Average Time on Market',
                    'Times Price Lowered', 'Average Price Decrease',
                    'Date Details Fetched', 'Image URLs', 'Events', 'Previous Days on Market',
                    'URL'
                ])

        with open('jsonData/config.json', 'r') as f:
            config = json.load(f)
        proxies = config['proxies']

        urls_to_scrape = []
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # Skip header
            except StopIteration:
                pass  # Empty file
            for row in reader:
                if row:  # Ensure row is not empty
                    url = row[1]  # Assuming link is in the 2nd column
                    if url not in scraped_urls:
                        urls_to_scrape.append(url)

        if not urls_to_scrape:
            logging.info("No new URLs to scrape. Waiting 60 seconds before checking again...")
            await asyncio.sleep(60)
            continue

        num_proxies = len(proxies)
        if num_proxies == 0:
            logging.error("No proxies found in config.json")
            return

        # Distribute URLs to proxies in a round-robin fashion
        proxy_url_map = {proxy: [] for proxy in proxies}
        for i, url in enumerate(urls_to_scrape):
            proxy = proxies[i % num_proxies]
            proxy_url_map[proxy].append(url)

        tasks = []
        for proxy, urls_chunk in proxy_url_map.items():
            if urls_chunk:
                task = asyncio.create_task(fetch_home_details(proxy, urls_chunk, start_time))
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)

        # Check if 15 minutes have elapsed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 900:  # 900 seconds = 15 minutes
            logging.info("15 minutes elapsed. Terminating browsers and restarting cycle.")
        else:
            logging.info(f"Cycle completed in {elapsed_time:.2f} seconds. Waiting before next cycle...")
            # Wait until 15 minutes from start if we haven't reached it yet
            remaining_time = max(0, 900 - elapsed_time)
            await asyncio.sleep(remaining_time)

        # Kill all chrome processes and wait 60 seconds
        logging.info("Terminating Chrome processes...")
        os.system('taskkill /F /IM chrome.exe /T')
        logging.info("Waiting 60 seconds before restart...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())