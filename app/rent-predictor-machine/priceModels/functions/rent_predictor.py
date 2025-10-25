from haversine import haversine, Unit
import pandas as pd
import numpy as np
import shap
import time

def preprocess_property_data(property_data, trained_columns, median_values, model, explainer):
    """
    Preprocesses a single property data dictionary, predicts the price,
    and calculates SHAP values using a robust, template-based encoding method.

    Args:
        property_data (dict): A dictionary containing the property's features.
        trained_columns (list): A list of columns from the training feature DataFrame (X). (Note: This is overridden by model's internal feature list for robustness).
        median_values (pd.Series): A pandas Series containing the median values of the training features (X.median()).
        model: The trained XGBoost model.
        explainer: The SHAP explainer object for the model.

    Returns:
        A tuple containing:
        - predicted_price (float): The predicted price for the property.
        - shap_values_dict (dict): A dictionary of SHAP values for the prediction.
        - preprocessed_df (pd.DataFrame): The preprocessed single-row DataFrame before dropping columns for prediction.
    """
    ENCODED_CSV_TEMPLATE = 'floridaHomeDataIncomeEncodedV2.csv'

    # --- ENCODING LOGIC ---
    try:
        encoded_template_df = pd.read_csv(ENCODED_CSV_TEMPLATE, nrows=0)
    except FileNotFoundError:
        print(f"Error: The template file '{ENCODED_CSV_TEMPLATE}' was not found.")
        raise
    
    df_single = pd.DataFrame(columns=encoded_template_df.columns.tolist(), index=[0])
    df_single.fillna(False, inplace=True)

    for key, value in property_data.items():
        if key in df_single.columns:
            # Convert "N/A" and other string representations of NaN to actual NaN
            if isinstance(value, str) and value.upper() in ['N/A', 'NA', 'NAN', 'NULL', '']:
                value = np.nan
            df_single.loc[0, key] = value

    appliances_str = property_data.get('Appliances')
    if pd.notna(appliances_str):
        appliances_list = [app.strip() for app in str(appliances_str).split(',')]
        for appliance in appliances_list:
            appliance_col = f'Appliance_{appliance}'
            if appliance_col in df_single.columns:
                df_single.loc[0, appliance_col] = True

    categorical_cols = ['Home Type', 'Cooling', 'Heating', 'Laundry', 'Parking']
    for cat_col in categorical_cols:
        value = property_data.get(cat_col)
        if pd.notna(value):
            encoded_col_name = f'{cat_col}_{value}'
            if encoded_col_name in df_single.columns:
                df_single.loc[0, encoded_col_name] = True
    
    # --- FIXED PREDICTION LOGIC ---
    # The source of truth for feature names is the model itself.
    # This is the most reliable way to prevent the mismatch error.
    model_feature_names = model.get_booster().feature_names

    # Create the DataFrame for prediction by reindexing the encoded data.
    # This crucial step adds any missing columns (e.g., 'Property Tax') with a fill value
    # and *removes* any extra columns (e.g., 'Price', 'Home Type', 'Cooling').
    df_for_prediction = df_single.reindex(columns=model_feature_names, fill_value=0)

    # Handle infinite values
    df_for_prediction.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert object columns that should be numeric to numeric types
    # First try to convert potential numeric object columns
    for col in df_for_prediction.columns:
        if df_for_prediction[col].dtype == 'object':
            try:
                # Try converting to numeric, if possible
                df_for_prediction[col] = pd.to_numeric(df_for_prediction[col], errors='coerce')
                print(f"Converted column {col} from object to numeric")
            except ValueError:
                # If numeric conversion fails, try converting to string
                df_for_prediction[col] = df_for_prediction[col].astype(str)
                print(f"Converted column {col} from object to string")
            except:
                pass  # Keep as object if conversion fails

    # Fill NaNs using median values. We only need medians for columns the model actually uses.
    median_values_for_model = median_values[median_values.index.isin(model_feature_names)]
    df_for_prediction.fillna(median_values_for_model, inplace=True)

    # Final check - ensure no remaining NaN values for numeric columns used by the model
    for col in model_feature_names:
        if col in df_for_prediction.columns and pd.isna(df_for_prediction[col].iloc[0]):
            # Try to fill with median from template data or a default
            if col in median_values_for_model.index:
                df_for_prediction[col].fillna(median_values_for_model[col], inplace=True)
            else:
                # Try to get median from the original encoding template
                try:
                    template_median = pd.read_csv(ENCODED_CSV_TEMPLATE, usecols=[col]).median().iloc[0]
                    df_for_prediction[col].fillna(template_median, inplace=True)
                    print(f"Used template median for column {col}: {template_median}")
                except Exception as e:
                    print(f"Error processing zipcode data: {e}")
                    # Last resort - fill with 0 for numeric columns, False for boolean
                    if df_for_prediction[col].dtype in ['float64', 'int64']:
                        df_for_prediction[col].fillna(0, inplace=True)
                        print(f"Filled column {col} with 0 as last resort")
                    else:
                        df_for_prediction[col].fillna(False, inplace=True)
                        print(f"Filled column {col} with False as last resort")
    
    # This line addresses the FutureWarning and correctly converts the Zipcode if it exists in the model
    if 'Zipcode' in df_for_prediction.columns and pd.notna(df_for_prediction.loc[0, 'Zipcode']):
        df_for_prediction['Zipcode'] = int(df_for_prediction['Zipcode'].iloc[0])

    # Predict the price using the now perfectly aligned DataFrame
    predicted_price = float(model.predict(df_for_prediction)[0])

    # Calculate SHAP values
    shap_values = explainer.shap_values(df_for_prediction)[0]
    shap_values_dict = dict(zip(df_for_prediction.columns, shap_values))

    # Return the original, more complete df_single as per the function's contract
    # Convert numpy types to Python float for JSON serialization
    return float(predicted_price), {k: float(v) for k, v in shap_values_dict.items()}, df_single


def find_nearby_properties(input_lat, input_lon, full_df, num_nearby=4):
    """
    Finds properties geographically close to the input property.

    Args:
        input_lat: Latitude of the input property.
        input_lon: Longitude of the input property.
        full_df: The full original DataFrame containing all properties.
        num_nearby: The number of nearby properties to find.

    Returns:
        A DataFrame containing data for the nearby properties.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = full_df.copy()

    # Calculate Haversine distance for each property
    distances = []
    input_coords = (input_lat, input_lon)

    df_copy_valid_coords = df_copy.dropna(subset=['Latitude', 'Longitude'])

    for index, row in df_copy_valid_coords.iterrows():
        prop_coords = (row['Latitude'], row['Longitude'])
        try:
            distance = haversine(input_coords, prop_coords, unit=Unit.MILES)
            distances.append((index, distance))
        except Exception:
            continue

    if not distances:
        return pd.DataFrame()
        

    distances_df = pd.DataFrame(distances, columns=['original_index', 'distance'])
    # Use the original index from the valid coordinates df to merge
    df_with_distances = df_copy_valid_coords.merge(distances_df, left_index=True, right_on='original_index')

    nearby_properties = df_with_distances.sort_values(by='distance').head(num_nearby)

    nearby_properties.head(3)

    return nearby_properties.drop(columns=['original_index', 'distance'])


def process_and_find_nearby(property_data, trained_columns, median_values, model, explainer, full_df, num_nearby):
    """
    Preprocesses a single property data dictionary, predicts the price,
    calculates SHAP values, and finds nearby properties.
    """
    predicted_price, shap_values_dict, preprocessed_df = preprocess_property_data(
        property_data, trained_columns, median_values, model, explainer
    )

    input_lat = float(property_data.get('Latitude'))
    input_lon = float(property_data.get('Longitude'))

    nearby_properties_df = pd.DataFrame() 

    nearby_properties_df = find_nearby_properties(
        input_lat, input_lon, full_df, num_nearby
    )

    return predicted_price, shap_values_dict, nearby_properties_df

def calculate_market_medians(df, property_data, zipcode, zipcode_data):
    """Calculate median market times by zipcode and for similar properties."""

    medians = {}

    # Median time on market by zipcode
    print("Using zipcode", zipcode)
    zipcode_filter = df['Zipcode'].astype(str).str.contains(str(zipcode), na=False)
    zipcode_data_from_df = df[zipcode_filter]

    medians['average_rent_prices'] = {}

    zipcode_data.to_csv('debug_zipcode_data.csv', index=False)
    
    zipcode_rows = zipcode_data[zipcode_data['Zipcode'].astype(str) == str(zipcode).strip()]
    
    zipcode_row = zipcode_rows
    print(f"Found zipcode data for {zipcode}")
    print(zipcode_row)

    for year in range(2020, 2026):
        col_name = 'Average_Rent_{}'.format(year)
        rent_value = zipcode_row[col_name].item()
        if pd.notna(rent_value) and rent_value != '' and str(rent_value).strip() != 'nan':
            medians['average_rent_prices'][str(year)] = float(rent_value)
        else:
            medians['average_rent_prices'][str(year)] = 0.0
      

    # Print the extracted data for debugging
    print(f"Extracted average_rent_prices: {medians['average_rent_prices']}")

    if not zipcode_data_from_df.empty and 'Most Recent Time on Market' in zipcode_data_from_df.columns:
        # Safe median calculation with empty check
        time_values = zipcode_data_from_df['Most Recent Time on Market'].dropna()
        if not time_values.empty:
            medians['median_time_on_market_by_zipcode'] = time_values.median()
        else:
            medians['median_time_on_market_by_zipcode'] = None
    else:
        medians['median_time_on_market_by_zipcode'] = None

    # Find 5 most similar properties and calculate their median
    if property_data:
        # Extract key features for similarity comparison
        features_to_compare = ['Price', 'Bedrooms', 'Bathrooms', 'Living Area']

        # Get property values for comparison
        prop_values = {}
        for feature in features_to_compare:
            if feature in property_data:
                value = property_data[feature]
                if pd.notna(value):
                    prop_values[feature] = value

        if prop_values:
            # Sample the dataframe to speed up similarity calculation
            if len(df) > 1000:
                df_sample = df.sample(n=1000, random_state=42)
            else:
                df_sample = df
            # Calculate similarity scores for properties in sample
            similarities = []
            for idx, row in df_sample.iterrows():
                similarity_score = 0
                valid_features = 0

                for feature, prop_value in prop_values.items():
                    if feature in row and pd.notna(row[feature]) and pd.notna(prop_value):
                        try:
                            # Ensure both values are numeric
                            row_value = pd.to_numeric(row[feature], errors='coerce')
                            prop_numeric_value = pd.to_numeric(prop_value, errors='coerce')

                            if pd.notna(row_value) and pd.notna(prop_numeric_value):
                                # Safe median calculation to avoid empty slice warning
                                feature_series = df[feature].dropna()
                                feature_median = feature_series.median() if not feature_series.empty else 1
                                if feature_median > 0 and feature_median != 0:  # Also check for zero median
                                    diff = abs(row_value - prop_numeric_value) / feature_median
                                    similarity_score += (1 - min(diff, 1))  # Higher score for lower difference
                                    valid_features += 1
                        except (TypeError, ValueError) as e:
                            # Skip this feature if conversion fails
                            continue

                if valid_features > 0:
                    similarities.append((idx, similarity_score / valid_features))

            # Get top 5 most similar
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_2_indices = [idx for idx, _ in similarities[:2]]

            if top_2_indices and 'Most Recent Time on Market' in df.columns:
                similar_days = df.loc[top_2_indices, 'Most Recent Time on Market'].dropna()
                medians['similar_houses'] = df.loc[top_2_indices].to_dict(orient='records')
                if not similar_days.empty:
                    medians['median_time_on_market_similar_properties'] = similar_days.median()
                else:
                    medians['median_time_on_market_similar_properties'] = None
            else:
                medians['median_time_on_market_similar_properties'] = None
        # Use zipcode_data if available to override or add additional metrics
        if zipcode_data is not None and zipcode is not None:
            try:
                zip_rows = zipcode_data[zipcode_data['Zipcode'] == zipcode]
                if not zip_rows.empty:
                    zip_row = zip_rows.iloc[0]
                    # Override median time with zipcode data if available
                    if 'Most Recent Time on Market' in zip_row.index:
                        medians['median_time_on_market_by_zipcode'] = zip_row['Most Recent Time on Market']
                    # Set cashflow from zipcode data
                    if 'Estimated Monthly Cash Flow' in zipcode_data.columns:
                        medians['estimated_monthly_cashflow'] = zip_row['Estimated Monthly Cash Flow']
                    # Set average rent prices from zipcode data for available years
                    pass
            except Exception as e:
                print(f"Error processing zipcode data: {e}")
        
            # Set defaults if not set
            if 'estimated_monthly_cashflow' not in medians:
                medians['estimated_monthly_cashflow'] = None
            if 'average_rent_prices' not in medians:
                medians['average_rent_prices'] = {}
        else:
            medians['median_time_on_market_similar_properties'] = None
    else:
        medians['median_time_on_market_similar_properties'] = None

    return medians

def generate_comprehensive_report(df, property_data, xgbr_rent, xgbr_days, xgbc_days, explainer_rent, zipcode_data, zipcode):
    """Generate complete report including rent prediction and days on market."""
    from .days_on_market import generate_days_on_market_data

    # Convert object columns to string before processing
    for col in property_data.keys():
        if isinstance(property_data[col], object):
            property_data[col] = str(property_data[col])

    # Get rent predictions and shap values
    predicted_price, shap_values_dict, nearby_properties_df = process_and_find_nearby(
        property_data, df.columns.tolist(), df.median(numeric_only=True), xgbr_rent, explainer_rent, df, 3
    )

    # Get days on market predictions
    try:
        days_report = generate_days_on_market_data(df, xgbr_days, xgbc_days, property_data)
    except Exception as e:
        print(f"Days on market prediction failed: {e}")
        days_report = {
            'predicted_time_on_market': None,
            'predicted_is_fast_sale': None,
            'predicted_fast_sale_probability': None,
            'shap_expected_value_regression': None
        }

    # Determine if it's "hot" based on fast sale probability
    is_hot = days_report.get('predicted_is_fast_sale') == 1 if days_report.get('predicted_is_fast_sale') is not None else None

    # Calculate market medians
    print("Pass in", zipcode)
    market_medians = calculate_market_medians(df, property_data, zipcode, zipcode_data)
    current_zip = zipcode
    if pd.notna(current_zip):
        try:
            current_zip = int(current_zip)
        except ValueError:
            current_zip = None

    # Find 4 closest zipcodes from zipcode_data using Haversine formula
    nearby_zipcodes = []
    if zipcode_data is not None and current_zip is not None:
        current_rows = zipcode_data[zipcode_data['Zipcode'] == current_zip]
        if not current_rows.empty:
            try:
                current_lat = current_rows.iloc[0]['Latitude_x']
                current_lon = current_rows.iloc[0]['Longitude_x']

                distances = []
                for idx, row in zipcode_data.iterrows():
                    if not pd.isna(row['Latitude_x']) and not pd.isna(row['Longitude_x']):
                        dist = haversine((current_lat, current_lon), (row['Latitude_x'], row['Longitude_x']), unit=Unit.MILES)
                        distances.append((row['Zipcode'], dist))

                # Sort by distance, take top 4 closest (excluding current)
                distances.sort(key=lambda x: x[1])
                nearby_zipcodes = [z for z, d in distances if z != current_zip][:4]
                print(f"Found {len(nearby_zipcodes)} nearby zipcodes: {nearby_zipcodes}")
            except Exception as e:
                print(f"Error calculating nearby zipcodes: {e}")
                nearby_zipcodes = []

    all_zipcodes = []
    if current_zip is not None:
        all_zipcodes.append(current_zip)
    all_zipcodes.extend(nearby_zipcodes)

    # Get median profits from zipcode data
    median_profits = {}
    if zipcode_data is not None and len(zipcode_data) > 0:
        for zip_code in all_zipcodes:
            zip_row = zipcode_data[zipcode_data['Zipcode'] == zip_code]
            if not zip_row.empty and 'Estimated Monthly Cash Flow' in zipcode_data.columns:
                median_profits[str(zip_code)] = zip_row.iloc[0]['Estimated Monthly Cash Flow']

    return {
        'predicted_price': predicted_price,
        'shap_values': shap_values_dict,
        'nearby_properties': nearby_properties_df.to_dict(orient='records'),
        'medians': df.median(numeric_only=True).to_dict(),
        'days_on_market': days_report.get('predicted_time_on_market'),
        'is_fast_sale': days_report.get('predicted_is_fast_sale'),
        'fast_sale_probability': days_report.get('predicted_fast_sale_probability'),
        'is_hot': is_hot,
        'market_medians': market_medians,
        'median_profits': median_profits
    }


def generate_rent_prediction_data(df, property_data, xgbr, explainer):
    """
    Top-level function to generate a full prediction package for a property.
    """
    # Convert "N/A" strings and other null representations to np.nan
    property_row_dict = {}
    for key, value in property_data.items():
        if isinstance(value, str) and value.upper() in ['N/A', 'NA', 'NAN', 'NULL', '']:
            property_row_dict[key] = np.nan
        else:
            property_row_dict[key] = value
    median_values = df.median(numeric_only=True)
    
    # The trained columns should be all columns from the processed dataframe `df`
    trained_columns = df.columns.tolist()

    # Impute missing lat/lon for the purpose of finding nearby properties if necessary
    if 'Latitude' not in property_row_dict or pd.isna(property_row_dict.get('Latitude')):
        property_row_dict['Latitude'] = df['Latitude'].median()
    if 'Longitude' not in property_row_dict or pd.isna(property_row_dict.get('Longitude')):
        property_row_dict['Longitude'] = df['Longitude'].median()

    predicted_price, shap_values_dict, nearby_properties = process_and_find_nearby(
        property_row_dict, trained_columns, median_values, xgbr, explainer, df
    )
    print(shap_values_dict)
    return {
        "predicted_price": predicted_price,
        "shap_values": shap_values_dict,
        "nearby_properties": nearby_properties.to_dict(orient='records'),
        #"medians": median_values.to_dict()
    }
