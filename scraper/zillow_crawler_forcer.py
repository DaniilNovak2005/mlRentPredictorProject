import subprocess
import time
import json
import pandas as pd
import pyarrow.parquet as pq
import os
import logging
import sys

# Constants
CRAWLER_SCRIPT = './zillow_crawler.py'
LOCATIONS_FILE = './jsonData/locations.json'
OUTPUT_FILE = './output.parquet'
RUN_INTERVAL_SECONDS = 1800  # 30 minutes

def setup_logging():
    """Sets up the logging configuration."""
    log_filename = f"{os.path.splitext(sys.argv[0])[0]}.log"
    with open(log_filename, 'w'):
        pass  # Clear the log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_total_locations():
    """Reads the total number of locations from the JSON file."""
    with open(LOCATIONS_FILE, 'r') as f:
        return json.load(f)['locations']

def get_processed_cities():
    """Gets the set of cities that have already been processed from the output file."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        table = pq.read_table(OUTPUT_FILE, columns=['City'])
        df = table.to_pandas()
        if not df.empty:
            return set(df['City'].unique())
    except Exception as e:
        logging.error(f"Could not read parquet file for caching: {e}")
    return set()

def terminate_processes():
    """
    Terminates conflicting Chrome and Python crawler processes using pkill for Linux compatibility.
    """
    logging.info("Terminating conflicting processes...")
    
    # Kill Chrome processes
    try:
        subprocess.run(['pkill', '-f', 'chrome'], check=False)
        logging.info("Attempted to terminate Chrome processes.")
    except Exception as e:
        logging.warning(f"Could not execute pkill for chrome: {e}")

    # Kill leftover Python crawler processes specifically
    try:
        # This targets only the crawler script to avoid killing the forcer itself.
        subprocess.run(['pkill', '-f', CRAWLER_SCRIPT], check=False)
        logging.info(f"Attempted to terminate Python processes for {CRAWLER_SCRIPT}.")
    except Exception as e:
        logging.warning(f"Could not execute pkill for python crawler: {e}")


def main():
    """Main function to run and manage the Zillow crawler."""
    setup_logging()
    
    total_locations = get_total_locations()
    total_cities = {loc['city'] for loc in total_locations}
    start = True
    while True:
        terminate_processes()
        if not start:
            logging.info("Waiting for 1 minute before starting the crawler...")
            time.sleep(60)
            start = False
        
        processed_cities = get_processed_cities()
        unprocessed_cities = total_cities - processed_cities
        
        if not unprocessed_cities:
            logging.info("All locations have been processed. Exiting.")
            break
            
        logging.info(f"{len(unprocessed_cities)} cities left to process.")
        logging.info(f"Starting {CRAWLER_SCRIPT}...")
        
        process = None
        try:
            # Start the crawler script as a subprocess
            process = subprocess.Popen(['python', CRAWLER_SCRIPT])
            
            # Wait for the specified interval
            logging.info(f"Waiting for {RUN_INTERVAL_SECONDS / 60} minutes...")
            time.sleep(RUN_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. Shutting down.")
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            if process:
                logging.info(f"Terminating {CRAWLER_SCRIPT} (PID: {process.pid}).")
                process.terminate()
                try:
                    # Wait a bit for the process to terminate gracefully
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # If it doesn't terminate, force kill it
                    logging.warning(f"Process did not terminate gracefully. Forcing kill.")
                    process.kill()
                logging.info("Process terminated. Restarting loop.")

if __name__ == "__main__":
    main()