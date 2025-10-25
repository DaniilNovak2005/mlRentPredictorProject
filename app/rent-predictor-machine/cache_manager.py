import pandas as pd
import os
import json
import hashlib
from datetime import datetime, timedelta

CACHE_DIR = 'cache'
ZILLOW_CACHE_FILE = os.path.join(CACHE_DIR, 'zillow_cache.parquet')
INCOME_CACHE_FILE = os.path.join(CACHE_DIR, 'income_cache.parquet')
CACHE_EXPIRE_DAYS = 30  # Data expires after 30 days

def ensure_cache_dir():
    """Ensure cache directory exists"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def generate_cache_key(link):
    return hashlib.md5(link.encode()).hexdigest()

def load_cache(cache_file):
    """Load cache from parquet file"""
    ensure_cache_dir()
    if os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            print(f"Loaded {len(df)} cached entries from {cache_file}")
            return df
        except Exception as e:
            print(f"Error loading cache from {cache_file}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(df, cache_file):
    """Save cache to parquet file"""
    ensure_cache_dir()
    try:
        df.to_parquet(cache_file, index=False)
        print(f"Saved {len(df)} entries to {cache_file}")
    except Exception as e:
        print(f"Error saving cache to {cache_file}: {e}")

def is_cache_expired(timestamp_str, expire_days=CACHE_EXPIRE_DAYS):
    """Check if cache entry is expired"""
    if not timestamp_str:
        return True

    try:
        # Parse the cached timestamp
        cached_time = pd.to_datetime(timestamp_str)

        # Check if it's expired
        expire_time = datetime.now() - timedelta(days=expire_days)
        return cached_time < expire_time
    except Exception:
        # If we can't parse the timestamp, consider it expired
        return True

def get_zillow_cached_data(cache_key):
    """Get cached zillow data if available and not expired"""
    df = load_cache(ZILLOW_CACHE_FILE)
    if df.empty:
        return None
    # Find entry by cache key
    entry = df[df['cache_key'] == cache_key]
    if not entry.empty:
        entry = entry.iloc[0].fillna("N/A")  # Get first match

        # Check if expired
        timestamp_str = entry.get('timestamp', '')
        if is_cache_expired(timestamp_str):
            print(f"Cache entry expired for key {cache_key}")
            return None

        # Convert cached data back to dict
        cached_data = entry.to_dict()

        # Remove cache metadata
        cache_metadata = ['cache_key', 'timestamp', 'zillow_link']
        for key in cache_metadata:
            cached_data.pop(key, None)

        print(f"Cache hit for zillow data: {cache_key}")
        return cached_data

    return None

def save_zillow_data(cache_key, data_dict, zillow_link=''):
    """Save zillow data to cache"""
    df = load_cache(ZILLOW_CACHE_FILE)

    # Add cache metadata
    data_to_save = data_dict.copy()
    for k, v in data_to_save.items():
        if v == 'N/A' or v is None:
            data_to_save[k] = pd.NA
    data_to_save['cache_key'] = cache_key
    data_to_save['timestamp'] = datetime.now().isoformat()
    data_to_save['zillow_link'] = zillow_link or data_dict.get('zillow_link', '')

    # Remove entry if it exists (to avoid duplicates)
    if not df.empty:
        df = df[df['cache_key'] != cache_key]

    # Add new entry
    new_row = pd.DataFrame([data_to_save])
    df = pd.concat([df, new_row], ignore_index=True)

    # Keep only recent entries (cleanup old entries)
    cleanup_cache(df, 'zillow')

    save_cache(df, ZILLOW_CACHE_FILE)
    print(f"Cache miss - saved zillow data for key: {cache_key}")
    import time
    time.sleep(20)

def get_income_cached_data(zipcode, state):
    """Get cached income data if available and not expired"""
    df = load_cache(INCOME_CACHE_FILE)
    if df.empty:
        return None

    # Find entry by zipcode and state
    entry = df[(df['zipcode'] == str(zipcode)) & (df['state'] == str(state))]
    if not entry.empty:
        entry = entry.iloc[0]  # Get first match

        # Check if expired
        timestamp_str = entry.get('timestamp', '')
        if is_cache_expired(timestamp_str):
            print(f"Cache entry expired for income data: {zipcode}, {state}")
            return None

        # Convert cached data back to dict
        cached_data = entry.to_dict()

        # Remove cache metadata
        cache_metadata = ['zipcode', 'state', 'timestamp']
        for key in cache_metadata:
            cached_data.pop(key, None)

        print(f"Cache hit for income data: {zipcode}, {state}")
        return cached_data

    return None

def save_income_data(zipcode, state, data_dict):
    """Save income data to cache"""
    df = load_cache(INCOME_CACHE_FILE)

    # Add cache metadata
    data_to_save = data_dict.copy()
    data_to_save['zipcode'] = str(zipcode)
    data_to_save['state'] = state
    data_to_save['timestamp'] = datetime.now().isoformat()

    # Remove entry if it exists (to avoid duplicates)
    if not df.empty:
        df = df[(df['zipcode'] != str(zipcode)) | (df['state'] != state)]

    # Add new entry
    new_row = pd.DataFrame([data_to_save])
    df = pd.concat([df, new_row], ignore_index=True)

    # Keep only recent entries
    cleanup_cache(df, 'income')

    save_cache(df, INCOME_CACHE_FILE)
    print(f"Cache miss - saved income data for: {zipcode}, {state}")

def cleanup_cache(df, cache_type, max_entries=1000):
    """Clean up old cache entries to prevent file from growing too large"""
    if df.empty:
        return df

    # Sort by timestamp (most recent first)
    df = df.sort_values('timestamp', ascending=False)

    # Keep only the most recent entries
    if len(df) > max_entries:
        df = df.head(max_entries)
        print(f"Cleaned up {cache_type} cache to {max_entries} entries")

    return df