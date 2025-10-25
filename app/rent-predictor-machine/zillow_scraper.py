import asyncio
import json
import nodriver as uc
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import statistics

BASE_URL = "https://www.zillow.com"
WAIT_TIME = 1.5


def calculate_price_history_stats(price_history):
    """
    Calculates detailed rental and sale statistics from Zillow's price history.
    """
    if not price_history:
        return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', '[]', 'N/A'

    try:
        sorted_history = sorted(price_history, key=lambda x: x.get('time', 0))
        
        # Capture all events for the 'Events' field
        events = []
        for event in sorted_history:
            event_data = {'type': event.get('event'), 'date': event.get('time')}
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
                        price_changes.append({'time': end_event.get('time'), 'price': end_event.get('price')})
                    elif end_event.get('event') == 'Listing removed':
                        end_time = end_event.get('time')
                        if start_time and end_time:
                            rental_periods.append({
                                'start_time': start_time, 'end_time': end_time,
                                'start_price': start_price, 'price_changes': price_changes,
                                'end_price': price_changes[-1]['price'] if price_changes else start_price
                            })
                        i = j
                        break
                    j += 1
            i += 1

        if not rental_periods:
            return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', json.dumps(events), 'N/A'

        # Calculate stats from all rental periods
        durations = [(datetime.fromtimestamp(p['end_time'] / 1000) - datetime.fromtimestamp(p['start_time'] / 1000)).days for p in rental_periods]
        
        most_recent_period = rental_periods[-1]
        most_recent_time_on_market = durations[-1] if durations else 'N/A'
        previous_days_on_market = durations[-2] if len(durations) > 1 else 'N/A'
        average_time_on_market = round(statistics.mean(durations), 2) if durations else 'N/A'
        
        price_lowered_count = 0
        price_decreases = []
        last_price = most_recent_period['start_price']
        for change in most_recent_period['price_changes']:
            if change.get('price') and last_price and change.get('price') < last_price:
                price_lowered_count += 1
                price_decreases.append(last_price - change['price'])
            last_price = change.get('price')
            
        average_price_decrease = round(statistics.mean(price_decreases), 2) if price_decreases else 0.0
        last_rental_price = most_recent_period['end_price']
        times_rented = len(rental_periods)
        
        last_rental_time = most_recent_period['end_time']
        years_since_rented = round((datetime.now() - datetime.fromtimestamp(last_rental_time / 1000)).days / 365.25, 2) if last_rental_time else 'N/A'
            
        return (last_rental_price, times_rented, years_since_rented,
                most_recent_time_on_market, average_time_on_market,
                price_lowered_count, average_price_decrease, json.dumps(events), previous_days_on_market)

    except (KeyError, TypeError, IndexError, statistics.StatisticsError):
        return 'N/A', 0, 'N/A', 'N/A', 'N/A', 0, 'N/A', '[]', 'N/A'


def calculate_tax_history_stats(tax_history):
    """Calculates financial statistics from Zillow's tax history."""
    if not tax_history or len(tax_history) < 2:
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'
    try:
        sorted_history = sorted(tax_history, key=lambda x: x.get('time', 0))
        ending_value = sorted_history[-1].get('value')
        beginning_value = sorted_history[0].get('value')
        
        if not all([ending_value, beginning_value, sorted_history[-1].get('time'), sorted_history[0].get('time')]):
            return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

        num_years = (datetime.fromtimestamp(sorted_history[-1]['time'] / 1000) - datetime.fromtimestamp(sorted_history[0]['time'] / 1000)).days / 365.25
        cagr = round(((ending_value / beginning_value) ** (1 / num_years)) - 1, 4) if beginning_value > 0 and num_years > 0 else 'N/A'

        value_increases = [h['valueIncreaseRate'] for h in sorted_history if 'valueIncreaseRate' in h]
        tax_increases = [h['taxIncreaseRate'] for h in sorted_history if 'taxIncreaseRate' in h]

        avg_value_increase = round(statistics.mean(value_increases), 4) if value_increases else 'N/A'
        avg_tax_increase = round(statistics.mean(tax_increases), 4) if tax_increases else 'N/A'
        std_dev_value_increase = round(statistics.stdev(value_increases), 4) if len(value_increases) > 1 else 0.0
        std_dev_tax_increase = round(statistics.stdev(tax_increases), 4) if len(tax_increases) > 1 else 0.0
        return cagr, avg_value_increase, avg_tax_increase, std_dev_value_increase, std_dev_tax_increase
    except (KeyError, TypeError, IndexError, statistics.StatisticsError):
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

def _extract_property_details(details_blob, soup):
    """Extracts and structures property details from the JSON blob and page soup."""
    if not details_blob:
        return None
    
    json.dump(details_blob, open('temp_details.json','w'))

    # Extract nested data objects
    tax_history = details_blob.get('taxHistory', [])
    price_history = details_blob.get('priceHistory', [])
    schools_data = details_blob.get('schools', [])
    reso_facts = details_blob.get('resoFacts', {})

    # Process data with helper functions
    (last_rental_price, times_rented, years_since_rented,
     most_recent_time_on_market, average_time_on_market,
     price_lowered_count, average_price_decrease,
     events_json, previous_days_on_market) = calculate_price_history_stats(price_history)

    (cagr, avg_val_inc, avg_tax_inc,
     std_val_inc, std_tax_inc) = calculate_tax_history_stats(tax_history)

    # School Info
    school_info = {'Primary': {}, 'Middle': {}, 'High': {}}
    for s in schools_data:
        level = s.get('level')
        if level in school_info and not school_info[level]:
            school_info[level] = {'distance': s.get('distance'), 'rating': s.get('rating')}
            
    # Image URLs
    image_urls = [p['url'] for p in details_blob.get('photos', []) if p.get('url')]

    # --- FIX STARTS HERE ---
    # Helper function to safely join features
    def safe_join(data):
        if data is None:
            return 'N/A'
        if isinstance(data, list):
            return ", ".join(data)
        return str(data) # If it's not a list, just convert to string

    appliances = safe_join(reso_facts.get('appliances'))
    cooling = safe_join(reso_facts.get('cooling'))
    heating = safe_join(reso_facts.get('heating'))
    laundry = safe_join(reso_facts.get('laundryFeatures'))
    parking = safe_join(reso_facts.get('parkingFeatures'))
    # --- FIX ENDS HERE ---

    # Assemble the final dictionary
    return {
        'Street Address': details_blob.get('streetAddress', 'N/A'),
        'City': details_blob.get('city', 'N/A'),
        'State': details_blob.get('state', 'N/A'),
        'Zipcode': details_blob.get('zipcode', 'N/A'),
        'Price': details_blob.get('price', 'N/A'),
        'Bedrooms': details_blob.get('bedrooms', 'N/A'),
        'Bathrooms': details_blob.get('bathrooms', 'N/A'),
        'Living Area': details_blob.get('livingAreaValue', 'N/A'),
        'Home Type': details_blob.get('homeType', 'N/A'),
        'Year Built': details_blob.get('yearBuilt', 'N/A'),
        'Property Tax': tax_history[0].get('taxPaid') if tax_history else 'N/A',
        'Property Tax Rate': details_blob.get('propertyTaxRate', 'N/A'),
        'Latitude': details_blob.get('latitude', 'N/A'),
        'Longitude': details_blob.get('longitude', 'N/A'),
        'Days on Zillow': details_blob.get('daysOnZillow', 'N/A'),
        'Last Rental Price': last_rental_price,
        'Times Rented': times_rented,
        'Years Since Rented': years_since_rented,
        'CAGR': cagr,
        'Avg Annual Value Increase': avg_val_inc,
        'Avg Annual Tax Increase': avg_tax_inc,
        'Std Dev Value Increase': std_val_inc,
        'Std Dev Tax Increase': std_tax_inc,
        'Walk Score': soup.find('a', {'aria-describedby': 'walk-score-text'}).text if soup.find('a', {'aria-describedby': 'walk-score-text'}) else 'N/A',
        'Transit Score': soup.find('a', {'aria-describedby': 'transit-score-text'}).text if soup.find('a', {'aria-describedby': 'transit-score-text'}) else 'N/A',
        'Bike Score': soup.find('a', {'aria-describedby': 'bike-score-text'}).text if soup.find('a', {'aria-describedby': 'bike-score-text'}) else 'N/A',
        'Primary School Distance': school_info.get('Primary', {}).get('distance', 'N/A'),
        'Primary School Rating': school_info.get('Primary', {}).get('rating', 'N/A'),
        'Middle School Distance': school_info.get('Middle', {}).get('distance', 'N/A'),
        'Middle School Rating': school_info.get('Middle', {}).get('rating', 'N/A'),
        'High School Distance': school_info.get('High', {}).get('distance', 'N/A'),
        'High School Rating': school_info.get('High', {}).get('rating', 'N/A'),
        'Appliances': appliances,
        'Cooling': cooling,
        'Heating': heating,
        'Laundry': laundry,
        'Parking': parking,
        'Zestimate': details_blob.get('zestimate', 'N/A'),
        'Rent Zestimate': details_blob.get('rentZestimate', 'N/A'),
        'Tax Assessed Value': details_blob.get('taxAssessedValue', 'N/A'),
        'Lot Area Value': details_blob.get('lotAreaValue', 'N/A'),
        'Most Recent Time on Market': most_recent_time_on_market,
        'Average Time on Market': average_time_on_market,
        'Times Price Lowered': price_lowered_count,
        'Average Price Decrease': average_price_decrease,
        'Date Details Fetched': datetime.now().strftime("%Y-%m-%d"),
        'Image URLs': json.dumps(image_urls),
        'Events': events_json,
        'Previous Days on Market': previous_days_on_market,
    }


async def fetch_home_details(browser, url):
    """
    Uses a browser to fetch Zillow home details for a single URL.
    """
    await asyncio.sleep(2)
    if not url.startswith("http"):
        full_url = BASE_URL + url
    else:
        full_url = url

    logging.info(f"Fetching URL: {full_url}")
    
    try:
        page = await browser.get(full_url)
        if not page:
            logging.warning(f"Failed to load page for {full_url}")
            return None
        await asyncio.sleep(10)
        # Wait for the script element to be loaded, then get content
        script_element = None
        if await page.select("#__NEXT_DATA__", timeout=60):
            page_content = await page.get_content()
            soup = BeautifulSoup(page_content, 'html.parser')
            script_element = soup.find('script', {'id': '__NEXT_DATA__'})
        
        if not script_element or not script_element.string:
            logging.warning(f"Could not find or extract text from '__NEXT_DATA__' script for {full_url}")
            return None

        try:
            logging.info("Attempting to parse __NEXT_DATA__ JSON...")
            json_data = json.loads(script_element.string)
            logging.info("Successfully parsed __NEXT_DATA__ JSON.")
        except json.JSONDecodeError:
            logging.error(f"JSON parsing error for {full_url}.")
            return None

        try:
            json.dump(json_data, open("debug_property_data.json", "w"), indent=2)  # For debugging purposes
            details = None

            # Path 1: Individual Home Details Page (HDP)
            # Data is in a stringified JSON inside a cache key.
            propsString = json_data['props']['pageProps']['componentProps'].get('gdpClientCache') or json_data['props']['pageProps'].get('apiCache')
            property_data = json.loads(propsString)
            json.dump(property_data, open("debug_property_data.json", "w"), indent=2)
            if property_data:
                logging.info("Found data in 'gdpClientCache' or 'apiCache'. Processing as HDP.")
                property_key = list(property_data.keys())[0]
                if property_key:
                    details = property_data.get(property_key, {}).get('property')
            
            # Path 2: Search Results Page
            # If the first path failed, check if it's a search results page.
            
            return _extract_property_details(details, soup)
        except (KeyError, IndexError, TypeError) as e:
            logging.error(f"Could not extract data for {full_url}. Error: {e}")
            return None

    except Exception as e:
        import traceback
        logging.error(f"An error occurred while processing {full_url}: {e}")
        traceback.print_exc()
        return None