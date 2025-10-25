import asyncio
import nodriver as uc
import pandas as pd
import numpy as np
import json
import os
import re
import logging
import sys
from bs4 import BeautifulSoup

# --- Caching ---
ZIPCODE_CACHE_FILE = 'zipcode_cache.json'

def load_zipcode_cache():
    if os.path.exists(ZIPCODE_CACHE_FILE):
        with open(ZIPCODE_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_zipcode_cache(cache):
    with open(ZIPCODE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

# --- State to URL Mapping ---
STATE_MAP = {
    'FL': 'florida'
    # Add other states here as needed
}

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
    state_url_path = STATE_MAP.get(state.upper())
    if not state_url_path:
        raise ValueError(f"State '{state}' not supported.")

    logging.info(f"Scraping data for zipcode: {zipcode} in {state}...")
    url = f"https://www.incomebyzipcode.com/{state_url_path}/{int(zipcode)}"
    page = await browser.get(url)
    data = {'zipcode': zipcode}

    try:
        await page.select('.img-fluid', timeout=15)
        content = await page.get_content()
        soup = BeautifulSoup(content, 'html.parser')

        def get_income_data(soup, header_text):
            header = soup.find('h2', string=lambda text: header_text in text if text else False)
            if header:
                table = header.find_next_sibling('table')
                if table:
                    cells = table.find_all('td')
                    if len(cells) > 1:
                        return clean_value(cells[1].get_text(strip=True))
            return None

        data['median_household_income'] = get_income_data(soup, 'Median Household Income')
        data['average_household_income'] = get_income_data(soup, 'Average Household Income')
        data['per_capita_income'] = get_income_data(soup, 'Per-Capita Income')

        high_income_header = soup.find('h2', string=lambda text: 'High Income Households' in text if text else False)
        if high_income_header:
            p = high_income_header.find_next_sibling('p')
            if p:
                match = re.search(r'(\d+\.\d+)%', p.get_text())
                if match:
                    data['high_income_households_percent'] = clean_value(match.group(1))

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
async def main(zipcode, state):
    log_filename = f"{sys.argv[0].split('.')[0]}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    zipcode_cache = load_zipcode_cache()
    
    if str(zipcode) in zipcode_cache:
        logging.info(f"Returning cached data for zipcode {zipcode}.")
        return zipcode_cache[str(zipcode)]

    browser = None
    scraped_data = None
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        proxies = config.get('proxies', [])
        
        browser_args = []
        if proxies:
            proxy = proxies[0] # Using the first proxy for simplicity
            browser_args.append(f'--proxy-server={proxy}')

        browser = await uc.start(browser_args=browser_args)
        scraped_data = await scrape_zipcode_data(browser, zipcode, state)
        
        if scraped_data:
            zipcode_cache[str(zipcode)] = scraped_data
            save_zipcode_cache(zipcode_cache)
            
    except Exception as e:
        logging.error(f"An error occurred in main workflow: {e}")
    finally:
        if browser:
            browser.stop()
            
    return scraped_data

if __name__ == '__main__':
    # Example usage: python income_scraper.py 33605 FL
    if len(sys.argv) != 3:
        print("Usage: python income_scraper.py <zipcode> <state_abbr>")
        sys.exit(1)
        
    zipcode_arg = sys.argv[1]
    state_arg = sys.argv[2]
    
    result = asyncio.run(main(zipcode_arg, state_arg))
    
    if result:
        print(json.dumps(result, indent=4))