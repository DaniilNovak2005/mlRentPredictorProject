import asyncio
import nodriver as uc
import pandas as pd
import numpy as np
import json
import os
import re
import logging
import sys
import csv
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# --- Configuration ---
INPUT_FILE = 'csvData/floridaHomeDetailsV2.csv'
OUTPUT_FILE = 'csvData/floridaHomeDetailsIncome.csv'
ZIPCODE_CACHE_FILE = 'jsonData/zipcode_cache.json'
BASE_URL = 'https://www.incomebyzipcode.com/'
STATES_MAP = {
    'AL': 'alabama', 'AK': 'alaska', 'AZ': 'arizona', 'AR': 'arkansas', 'CA': 'california',
    'CO': 'colorado', 'CT': 'connecticut', 'DE': 'delaware', 'FL': 'florida', 'GA': 'georgia',
    'HI': 'hawaii', 'ID': 'idaho', 'IL': 'illinois', 'IN': 'indiana', 'IA': 'iowa',
    'KS': 'kansas', 'KY': 'kentucky', 'LA': 'louisiana', 'ME': 'maine', 'MD': 'maryland',
    'MA': 'massachusetts', 'MI': 'michigan', 'MN': 'minnesota', 'MS': 'mississippi', 'MO': 'missouri',
    'MT': 'montana', 'NE': 'nebraska', 'NV': 'nevada', 'NH': 'new hampshire', 'NJ': 'new jersey',
    'NM': 'new mexico', 'NY': 'new york', 'NC': 'north carolina', 'ND': 'north dakota', 'OH': 'ohio',
    'OK': 'oklahoma', 'OR': 'oregon', 'PA': 'pennsylvania', 'RI': 'rhode island', 'SC': 'south carolina',
    'SD': 'south dakota', 'TN': 'tennessee', 'TX': 'texas', 'UT': 'utah', 'VT': 'vermont',
    'VA': 'virginia', 'WA': 'washington', 'WV': 'west virginia', 'WI': 'wisconsin', 'WY': 'wyoming',
    'DC': 'district of columbia'
}

# --- Caching ---
def load_zipcode_cache():
    if os.path.exists(ZIPCODE_CACHE_FILE):
        with open(ZIPCODE_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_zipcode_cache(cache):
    with open(ZIPCODE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

# --- Scraping ---
def clean_value(value_str):
    if not isinstance(value_str, str) or value_str == '$-1':
        return np.nan
    cleaned_str = re.sub(r'[$,%]', '', value_str)
    try:
        return float(cleaned_str)
    except (ValueError, TypeError):
        return np.nan

async def scrape_zipcode_data(browser, zipcode, state):
    logging.info(f"Scraping data for zipcode: {zipcode} in {state}...")
    state_name = STATES_MAP.get(state.upper())
    if not state_name:
        logging.error(f"Invalid state abbreviation: {state}")
        return None
    url = f"{BASE_URL}{state_name}/{int(zipcode)}"
    page = await browser.get(url)
    data = {'zipcode': zipcode}

    try:
        # Wait for the first key element to appear
        await page.select('.img-fluid', timeout=15)
        
        # Get page content and parse with BeautifulSoup
        content = await page.get_content()
        soup = BeautifulSoup(content, 'html.parser')

        # Helper function to extract data from a section
        def get_income_data(soup, header_text):
            header = soup.find('h2', string=lambda text: header_text in text if text else False)
            if header:
                table = header.find_next_sibling('table')
                if table:
                    value_cell = table.find_all('td')[1]
                    if value_cell:
                        return clean_value(value_cell.get_text(strip=True))
            return None

        data['median_household_income'] = get_income_data(soup, 'Median Household Income')
        data['average_household_income'] = get_income_data(soup, 'Average Household Income')
        data['per_capita_income'] = get_income_data(soup, 'Per-Capita Income')

        # High Income Households
        high_income_header = soup.find('h2', string=lambda text: 'High Income Households' in text if text else False)
        if high_income_header:
            p = high_income_header.find_next_sibling('p')
            if p:
                match = re.search(r'(\d+\.\d+)%', p.get_text())
                if match:
                    data['high_income_households_percent'] = clean_value(match.group(1))

        # Median Household Income by Age
        age_income_header = soup.find('h2', string=lambda text: 'Median Household Income by Age' in text if text else False)
        if age_income_header:
            table_container = age_income_header.find_next_sibling('div', class_='table-responsive')
            if table_container:
                table = table_container.find('table')
                if table:
                    rows = table.find_all('tr')
                    if len(rows) > 1:
                        headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]
                        values = [td.get_text(strip=True) for td in rows[1].find_all('td')]
                        for i in range(1, len(headers)):
                            data[f'median_income_{headers[i].replace(" ", "_").lower()}'] = clean_value(values[i])
        
        # Remove None/NaN values and check if we got any data
        cleaned_data = {k: v for k, v in data.items() if pd.notna(v)}

        if len(cleaned_data) > 1:
            logging.info(f"Successfully scraped data for {zipcode}")
            return cleaned_data
        else:
            logging.warning(f"No data found for zipcode {zipcode}, but page loaded.")
            return None

    except Exception as e:
        logging.error(f"Error scraping {zipcode}: {e}")
        return None


# --- Main Workflow ---
async def process_zipcode_worker(proxy, zipcodes_to_process, results_dict):
    logging.info(f"Worker with proxy {proxy} starting for {len(zipcodes_to_process)} zipcodes.")
    browser = None
    try:
        browser = await uc.start(browser_args=[f'--proxy-server={proxy}'])
        for zipcode, state in zipcodes_to_process:
            scraped_data = await scrape_zipcode_data(browser, zipcode, state)
            if scraped_data:
                results_dict[zipcode] = scraped_data
    except Exception as e:
        logging.error(f"Error in worker with proxy {proxy}: {e}")
    finally:
        if browser:
            browser.stop()

async def main():
    # Set up logging
    log_filename = f"{sys.argv[0].split('.')[0]}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
        return

    input_df = pd.read_csv(INPUT_FILE)
    
    zip_state_df = input_df[['Zipcode', 'State']].dropna().drop_duplicates(subset=['Zipcode', 'State'])
    
    zipcode_cache = load_zipcode_cache()
    
    # Get unique zipcodes from the input file
    unique_zipcodes_in_file = zip_state_df['Zipcode'].unique()
    
    # Determine which zipcodes are not in the cache
    zipcodes_to_scrape = [z for z in unique_zipcodes_in_file if str(z) not in zipcode_cache]
    logging.info(f"Found {len(zipcodes_to_scrape)} new zipcodes to scrape.")

    if zipcodes_to_scrape:
        # Get the (zipcode, state) pairs for the zipcodes we need to scrape.
        # If a zipcode is associated with multiple states, we'll just use the first one.
        pairs_to_scrape_df = zip_state_df[zip_state_df['Zipcode'].isin(zipcodes_to_scrape)]
        unique_pairs_to_scrape = list(pairs_to_scrape_df.drop_duplicates(subset=['Zipcode']).to_records(index=False))

        with open('jsonData/config.json', 'r') as f:
            config = json.load(f)
        proxies = config['proxies']

        if not proxies:
            logging.error("No proxies found in config.json")
            return

        scraped_results = {}
        num_proxies = len(proxies)
        proxy_zipcode_map = {proxy: [] for proxy in proxies}
        for i, (zipcode, state) in enumerate(unique_pairs_to_scrape):
            proxy = proxies[i % num_proxies]
            proxy_zipcode_map[proxy].append((zipcode, state))

        tasks = []
        for proxy, zipcode_chunk in proxy_zipcode_map.items():
            if zipcode_chunk:
                task = asyncio.create_task(process_zipcode_worker(proxy, zipcode_chunk, scraped_results))
                tasks.append(task)
            
        await asyncio.gather(*tasks)

        zipcode_cache.update(scraped_results)
        save_zipcode_cache(zipcode_cache)

    # Merge data
    income_df = pd.DataFrame.from_dict(zipcode_cache, orient='index')
    
    # Ensure zipcode columns are of the same type for merging
    input_df['Zipcode'] = input_df['Zipcode'].astype(str)
    income_df.index = income_df.index.astype(str)

    merged_df = input_df.merge(income_df, left_on='Zipcode', right_index=True, how='left')
    
    merged_df.to_csv(OUTPUT_FILE, index=False)

    logging.info(f"Processing complete. Results saved to {OUTPUT_FILE}.")

if __name__ == '__main__':
    asyncio.run(main())