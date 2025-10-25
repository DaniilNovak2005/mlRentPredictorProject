from flask import Flask, request, jsonify
import xgboost as xgb
import os
import json
import asyncio
import nodriver as uc
import random
import pandas as pd
import numpy as np
import shap
import re
import traceback
from zillow_scraper import fetch_home_details
from income_scraper import main as income_scraper_main
from predictor import predict_new_house_price_combined
from priceModels.functions.rent_predictor import generate_comprehensive_report
from priceModels.functions.calculate_insurance import calculate as calculate_insurance

from cache_manager import (
    get_zillow_cached_data,
    save_zillow_data,
    get_income_cached_data,
    save_income_data,
    generate_cache_key
)

app = Flask(__name__)

# Caching for models and dataframes
models_cache = {}
dataframes_cache = {}
medians_cache = {}
explainers_cache = {}
zipcode_data_cache = {}
insurance_data_cache = {}

def load_all_models():
    price_models_dir = 'priceModels'
    price_data_dir = 'priceData'
    if not os.path.exists(price_models_dir):
        return
    for category in os.listdir(price_models_dir):
        category_path = os.path.join(price_models_dir, category)
        if not os.path.isdir(category_path) or category == 'functions':
            continue
        # Load dataframe from priceData if exists
        csv_path = os.path.join(price_data_dir, f'{category}.csv')
        if not os.path.exists(csv_path):
            print(f"No CSV data found for category {category}, skipping...")
            continue
        df = pd.read_csv(csv_path)
        dataframes_cache[category] = df
        # Safe median calculation to avoid empty slice warnings
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        medians = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if not col_data.empty:
                medians[col] = col_data.median()
            else:
                medians[col] = 0  # Default median for empty columns
        medians_cache[category] = medians
        print(f"Loaded dataframe and medians for category: {category}")

        insurance_csv_path = os.path.join(price_data_dir, f'{category}InsuranceData.csv')
        if  os.path.exists(insurance_csv_path):
            insurance_df = pd.read_csv(insurance_csv_path)
            insurance_data_cache[category] = insurance_df
            print(f"Loaded insurance data for category: {category}")

        # Load zipcode median data if exists
        zipcode_csv_path = os.path.join(price_data_dir, f'{category}ZipcodeMedian.csv')
        if os.path.exists(zipcode_csv_path):
            zipcode_df = pd.read_csv(zipcode_csv_path)
            zipcode_data_cache[category] = zipcode_df
            print(f"Loaded zipcode data for category: {category}")
            print(f"Zipcode data columns: {zipcode_df.columns.tolist()}")
            print(f"Zipcode data shape: {zipcode_df.shape}")
            if len(zipcode_df) > 0:
                print(f"Sample row: {zipcode_df.iloc[0].to_dict()}")
        else:
            print(f"Warning: Zipcode file not found at {zipcode_csv_path}")
            print(f"Available files in {price_data_dir}: {[f for f in os.listdir(price_data_dir) if '.csv' in f]}")

        for file in os.listdir(category_path):
            if file.endswith('.json'):
                model_path = os.path.join(category_path, file)
                model_key = file.replace('.json', '')
                if 'binary' in file.lower():
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                else:
                    model = xgb.XGBRegressor()
                    model.load_model(model_path)
                models_cache[model_key] = model
                explainers_cache[model_key] = shap.TreeExplainer(model)
                print(f"Loaded model and explainer: {model_key}")

def extract_zipcode_from_url(url):
    # Find the path part
    match = re.search(r'/homedetails/([^/]+)/', url)
    if match:
        address_part = match.group(1)
        print(f"Address part from URL: {address_part}")
        # Find the last 5 digits before end
        zip_match = re.search(r'(\d{5})$', address_part.replace('-', ''))
        if zip_match:
            print(f"Zipcode found: {zip_match.group(1)}")
            return zip_match.group(1)
        else:
            print("No 5-digit zipcode pattern found")
    else:
        print("No homedetails path found in URL")
    return None

# Load all models at startup
load_all_models()

def get_model_and_df(state):
    if state in models_cache and state in dataframes_cache:
        return models_cache[state], dataframes_cache[state]

    df_path = os.path.join('priceData')
    model_path_folder = os.path.join("priceModels",state)
    # Find the first JSON model file in the directory
    try:
        model_file = next(f for f in os.listdir(model_path_folder) if f.endswith('.json'))
        model_path = os.path.join(model_path_folder, model_file)
    except StopIteration:
        raise FileNotFoundError(f"No JSON model file found for state '{state}' in {df_path}")

    # Find the first CSV file in the directory
    try:
        csv_file = next(f for f in os.listdir(df_path) if f.endswith('.csv') and state in f)
        df_full_path = os.path.join(df_path, csv_file)
    except StopIteration:
        raise FileNotFoundError(f"No CSV dataframe found for state '{state}' in {df_path}")

    print(model_path)

    model = xgb.Booster()
    model.load_model(model_path)
    
    df = pd.read_csv(df_full_path)

    models_cache[state] = model
    dataframes_cache[state] = df
    
    return model, df
    

async def scrape_with_random_proxy(link):
    browser = None
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        proxies = config.get('proxies', [])
        if not proxies:
            raise Exception("No proxies found in config.json")
        
        proxy = random.choice(proxies)
        print(f"Using proxy: {proxy}")
        
        browser = await uc.start() #browser_args=[f"--proxy-server=http://{proxy}"]
        details = await fetch_home_details(browser, link)
        return details
    finally:
        if browser:
            browser.stop()

@app.route('/create-report', methods=['POST'])
def create_report():
    print("--- New Create Report Request ---")
    data = request.get_json()
    state = data.get('state')
    details = data.get('details')
    link = data.get('link')
    settings = data.get('settings')
    print(data)
    if not link:
        link = details.get('link') if details else None

    propertyLink = link

    print(f"State: {state}, Has Details: {details is not None}, Has Link: {link is not None}")

    if not state:
        print("Error: State not provided.")
        return jsonify({'error': 'Please provide a state'}), 400
    zipcode_df = None
    try:
        print("Loading models, dataframe, medians, and explainers...")
        model_key_rent = state.lower() + 'PriceModel'
        model_key_days = state.lower() + 'DaysModel'
        model_key_days_classifier = state.lower() + 'BinaryModel'

        # Load rent models as before
        model_rent = models_cache.get(model_key_rent)
        df = dataframes_cache.get(state)
        explainer_rent = explainers_cache.get(model_key_rent)
        medians = medians_cache.get(state)

        zipcode_df = zipcode_data_cache.get(state)
        insurance_df = insurance_data_cache.get(state)

        # Load days-on-market models
        model_days_regressor = models_cache.get(model_key_days)  # flDaysModel
        model_days_classifier = models_cache.get(model_key_days_classifier)  # Classification model for is_fast_sale

        print(f"Rent Model: {model_rent is not None} (key: {model_key_rent})")
        print(f"Days Regressor: {model_days_regressor is not None} (key: {model_key_days})")
        print(f"Days Classifier: {model_days_classifier is not None} (key: {model_key_days_classifier})")
        print(f"DataFrame: {df is not None}, Rent Explainer: {explainer_rent is not None}, Medians: {medians is not None}")
        if df is not None:
            print(f"DataFrame shape: {df.shape}, columns: {len(df.columns)}")
        if zipcode_df is not None:
            print(f"Zipcode data shape: {zipcode_df.shape}")
        else:
            print("No zipcode data available")

        if not all([model_rent is not None, df is not None, medians is not None]):
            raise FileNotFoundError(f"No complete model data found for state '{state}'")

        if model_days_regressor is None:
            print("Warning: Days-on-market regression model not found, will proceed with rent prediction only")

        print("Models, dataframe, medians, and explainers loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model data: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"Unexpected error during model setup: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({'error': f'Model setup failed: {str(e)}'}), 500

    if not details and not link:
        print("Error: No details or Zillow link provided.")
        return jsonify({'error': 'Please provide features (details) or a Zillow link'}), 400

    if link:
        try:
            print(f"Checking cache for Zillow link: {link}")

            # Generate cache key from link
            cache_key = generate_cache_key(link)
            print(f"Generated cache key: {cache_key}")
            # Check cache first
            details = get_zillow_cached_data(cache_key)

            if details:
                print("Using cached zillow data")
            else:
                print("Cache miss - scraping Zillow link")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                details = loop.run_until_complete(scrape_with_random_proxy(link))
                loop.close()

            if details != None:
                print("Zillow scraping successful, saving to cache")
                save_zillow_data(cache_key, details, link)
            else:
                print("Error: Failed to scrape Zillow link, no details returned.")
                return jsonify({'error': 'Failed to scrape Zillow link'}), 500

        except Exception as e:
            print(f"An error occurred during scraping: {e}")
            return jsonify({'error': f"Scraping failed: {str(e)}"}), 500
    
    # Handle zipcode extraction from link if needed
    if link and ('Zipcode' not in details or not details.get('Zipcode') or details['Zipcode'] == "N/A"):
        extracted_zipcode = extract_zipcode_from_url(link)
        if extracted_zipcode:
            details['Zipcode'] = extracted_zipcode
            print(f"Extracted zipcode from URL: {extracted_zipcode}")

    if data.get('details'):
        for k, v in (data.get('details').items()):
            if v != None and v != 'N/A':
                details[k] = v

    # Run income scraper with caching (optional - don't fail if no zipcode)
    zipcode = details.get('Zipcode')
    income_data = None

    if zipcode and zipcode != 'N/A' and str(zipcode).strip():
        try:
            print(f"Starting income scraper process for zipcode {zipcode}...")
            # Check cache first
            income_data = get_income_cached_data(zipcode, state)

            if income_data:
                print(f"Using cached income data for zipcode {zipcode}")
            else:
                print(f"Cache miss - scraping income data for zipcode {zipcode}")
                income_data = asyncio.run(income_scraper_main(zipcode, state))

            if income_data:
                details.update(income_data)
                print(f"Income scraper returned {len(income_data)} fields:")
                for key, value in income_data.items():
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"Warning: Income scraping failed for zipcode {zipcode}: {e}")
            print("Continuing report generation with available data...")
            income_data = None
    else:
        print(f"Skipping income scraping - no valid zipcode found: {zipcode}")

    try:
        print("Creating comprehensive report...")

        # Log available data for debugging
        available_fields = list(details.keys()) if details else []
        print(f"Available data fields: {available_fields}")
        print(f"Number of available fields: {len(available_fields)}")

        # Store original scraped details before filtering
        scraped_details = details.copy() if details else {}

        # Remove fields not in training data
        exclude_fields = ['Parking', 'Price', 'Home Type', 'Heating', 'Cooling', 'Laundry', 'Zipcode']
        filt_details = {k: v for k, v in details.items() if k not in exclude_fields}
        zipcode = details.get('Zipcode')
        print(f"Zipcode before generate_comprehensive_report: {zipcode}")

        # Try to create comprehensive report, fall back to rent-only if days models unavailable
        try:
            # Create the comprehensive report with both rent and days predictions
            report = generate_comprehensive_report(
                df=df,
                property_data=filt_details,
                xgbr_rent=model_rent,
                xgbr_days=model_days_regressor,
                xgbc_days=model_days_classifier,
                explainer_rent=explainer_rent,
                zipcode_data=zipcode_df,
                zipcode=zipcode
            )
        except Exception as report_error:
            print(f"Comprehensive report creation failed: {report_error}")
            # Try to fall back to basic rent prediction only
            print("Attempting fallback to rent prediction only...")
            try:
                # Import the basic predictor function
                from predictor import predict_new_house_price_combined

                # Get basic prediction
                rent_prediction = predict_new_house_price_combined(df, model_rent, filt_details)

                # Create minimal report structure
                report = {
                    'predicted_price': rent_prediction,
                    'shap_values': {},
                    'market_medians': medians or {},
                    'nearby_properties': []
                }

                if explainer_rent:
                    # Try to get basic SHAP values
                    try:
                        # Convert filtered details to proper format for SHAP
                        input_df = pd.DataFrame([filt_details])
                        # Fill missing columns with medians
                        for col in df.columns:
                            if col not in input_df.columns and col in medians:
                                input_df[col] = medians[col]

                        shap_values = explainer_rent.shap_values(input_df.iloc[0].values.reshape(1, -1))

                        # Create simple SHAP dict (this is simplified)
                        shap_dict = {}
                        feature_names = input_df.columns.tolist()
                        for i, feature in enumerate(feature_names[:10]):  # Limit to first 10 most important
                            if hasattr(shap_values, '__len__') and len(shap_values) > 0:
                                if isinstance(shap_values, list):
                                    shap_dict[feature] = float(shap_values[0][i]) if len(shap_values[0]) > i else 0
                                else:
                                    shap_dict[feature] = float(shap_values[i]) if len(shap_values) > i else 0
                            else:
                                shap_dict[feature] = 0

                        report['shap_values'] = shap_dict
                    except Exception as shap_error:
                        print(f"SHAP calculation failed, using empty dict: {shap_error}")
                        report['shap_values'] = {}
                else:
                    print("No SHAP explainer available, using empty SHAP values")
                    report['shap_values'] = {}

                print("Fallback report created successfully")

            except Exception as fallback_error:
                print(f"Both comprehensive and fallback report generation failed: {fallback_error}")
                report = None
        
        insurance_info = None
        if insurance_df is not None and zipcode is not None:
            insurance_row = insurance_df[insurance_df['Zipcode'] == int(zipcode)]
            if not insurance_row.empty:
                insurance_info = insurance_row.iloc[0].to_dict()
                print(f"Found insurance data for zipcode {zipcode}: {insurance_info}")
            else:
                print(f"No insurance data found for zipcode {zipcode}")

        if insurance_info:
            insurance_result = calculate_insurance(details, settings, insurance_data_cache[state])
        else:
            insurance_result = None

        # Remove NaN values and replace with None for JSON serialization
        def clean_nan(obj):
            # Handle pandas/numpy NaN values
            if isinstance(obj, float) and (pd.isna(obj) or str(obj) == 'nan'):
                return None
            # Handle numpy data types that aren't JSON serializable
            elif hasattr(obj, 'dtype'):  # Check if it's a numpy scalar/array
                return clean_numpy_types(obj)
            elif isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            else:
                return obj

        def clean_numpy_types(obj):
            """Convert numpy types to Python native types"""
            try:
                import numpy as np
                # Handle numpy scalars (float32, float64, int32, int64, etc.)
                if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
                    return obj.item()  # Converts numpy scalar to Python type
                # Handle numpy arrays
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                # Handle pandas Series
                elif hasattr(obj, 'values'):
                    return obj.values.tolist() if hasattr(obj, 'tolist') else list(obj.values)
                else:
                    return str(obj)  # Fallback for other numpy objects
            except ImportError:
                return str(obj)  # Fallback if numpy not available

        cleaned_report = clean_nan(report)

        # Add scraped details to the response
        if cleaned_report:
            cleaned_report['scraped_details'] = clean_nan(scraped_details)
        elif scraped_details:
            # If report generation failed but we have scraped details, return them anyway
            cleaned_report = {'scraped_details': clean_nan(scraped_details)}

        cleaned_report['property_link'] = propertyLink
        if insurance_result:
            cleaned_report['insurance_info'] = insurance_result

        print(f"Final report contains {len(cleaned_report)} top-level keys:")
        print(f"Report keys: {list(cleaned_report.keys())}")
        if 'scraped_details' in cleaned_report and cleaned_report['scraped_details']:
            print(f"Scraped details contains {len(cleaned_report['scraped_details'])} fields:")
            print(f"Scraped details keys: {list(cleaned_report['scraped_details'].keys())}")

        if cleaned_report:
            print(f"Report generation successful")
            return jsonify(cleaned_report)
        else:
            print("No report could be generated")
            return jsonify({'error': 'Failed to generate report - no valid data available'}), 500

    except Exception as e:
        print(f"An error occurred during report creation: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({
            'error': f"Report creation failed: {str(e)}",
            'details': 'Please check your input data and try again'
        }), 500


if __name__ == '__main__':
    app.run(debug=True)