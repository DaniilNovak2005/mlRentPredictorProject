import pandas as pd
import numpy as np
import xgboost as xgb
import json
import traceback

def predict_house_outcomes(house_row, regression_model, classification_model, df_shuffled):
    """
    Predicts the 'is_fast_sale' and 'Most Recent Time on Market' for a single house row
    using trained classification and regression models.

    Args:
        house_row (pd.Series or pd.DataFrame): A row representing a single house with features.
        regression_model: The trained XGBoost Regressor model.
        classification_model: The trained XGBoost Classifier model.

    Returns:
        dict: A dictionary containing the predicted 'is_fast_sale',
              'Most Recent Time on Market', and the predicted probability of 'is_fast_sale'.
    """
    try:
        # Convert the input row to a DataFrame
        if isinstance(house_row, pd.Series):
            house_df = house_row.to_frame().T
        elif isinstance(house_row, pd.DataFrame):
            house_df = house_row.copy()
        elif isinstance(house_row, dict):
            house_df = pd.DataFrame([house_row])
        else:
            raise TypeError("Input 'house_row' must be a pandas Series, DataFrame, or dictionary.")

        # Define X_regression and X_classification from the training data used in previous cells
        # This ensures the prediction function has access to the training feature sets for column alignment and median filling
        global X_regression, X_classification # Declare as global to access the variables from the global scope
        columns_to_drop_regression = ['Most Recent Time on Market', 'Street Address', 'City', 'State', 'URL', 'Appliances', "Image URLs", "Date Details Fetched", "Events", 'is_fast_sale']
        columns_to_drop_classification = ['Most Recent Time on Market', 'Price', 'Street Address', 'City', 'State', 'URL', 'Appliances', "Image URLs", "Date Details Fetched", "Events", 'is_fast_sale']

        X_regression = df_shuffled.drop(columns=[col for col in columns_to_drop_regression if col in df_shuffled.columns])
        X_regression = X_regression.select_dtypes(include=np.number)
        X_regression.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Skip adding appliance columns - we'll remove missing ones from input data instead

        # Add calculated columns if not present
        if 'Rent Estimate to actual Price' not in X_regression.columns:
            X_regression['Rent Estimate to actual Price'] = 1.0  # Default ratio
        if 'price to area average by zipcode' not in X_regression.columns:
            X_regression['price to area average by zipcode'] = 100.0  # Default price/area
        if 'month listed' not in X_regression.columns:
            X_regression['month listed'] = 1  # Default month

        # Safe median filling to avoid empty slice warnings
        for col in X_regression.columns:
            col_data = X_regression[col].dropna()
            if not col_data.empty:
                X_regression[col] = X_regression[col].fillna(col_data.median())
            else:
                X_regression[col] = X_regression[col].fillna(0)  # Fallback for empty columns

        X_classification = df_shuffled.drop(columns=[col for col in columns_to_drop_classification if col in df_shuffled.columns])
        X_classification = X_classification.select_dtypes(include=np.number)
        X_classification.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Add ALL appliance columns from BOTH regression and classification models' feature names
        all_model_appliance_features = set()

        if hasattr(regression_model, 'get_booster'):
            regression_appliances = [col for col in regression_model.get_booster().feature_names if col.startswith('Appliance_')]
            all_model_appliance_features.update(regression_appliances)

        if hasattr(classification_model, 'get_booster'):
            classification_appliances = [col for col in classification_model.get_booster().feature_names if col.startswith('Appliance_')]
            all_model_appliance_features.update(classification_appliances)

        # Add all appliance columns from both models to both training datasets
        for col in all_model_appliance_features:
            if col not in X_regression.columns:
                X_regression[col] = 0
            if col not in X_classification.columns:
                X_classification[col] = 0

        if 'Rent Estimate to actual Price' not in X_classification.columns:
            X_classification['Rent Estimate to actual Price'] = 1.0
        if 'price to area average by zipcode' not in X_classification.columns:
            X_classification['price to area average by zipcode'] = 100.0
        if 'month listed' not in X_classification.columns:
            X_classification['month listed'] = 1

        # Safe median filling for classification model too
        for col in X_classification.columns:
            col_data = X_classification[col].dropna()
            if not col_data.empty:
                X_classification[col] = X_classification[col].fillna(col_data.median())
            else:
                X_classification[col] = X_classification[col].fillna(0)  # Fallback for empty columns


        # Handle 'Appliances' one-hot encoding
        training_columns_regression = X_regression.columns
        training_columns_classification = X_classification.columns

        # Combine all unique appliance columns from both training sets
        all_appliances_cols = sorted(list(set([col for col in training_columns_regression if col.startswith('Appliance_')] +
                                            [col for col in training_columns_classification if col.startswith('Appliance_')])))


        if 'Appliances' in house_df.columns and isinstance(house_df['Appliances'].iloc[0], str):
            appliances_list = [appliance.strip() for appliance in house_df['Appliances'].iloc[0].split(',')]
        else:
            appliances_list = []

        # Remove appliance columns that are not in the training data (since they're not critical)
        appliance_cols_in_house = [col for col in house_df.columns if col.startswith('Appliance_')]
        appliance_cols_in_training = [col for col in training_columns_regression if col.startswith('Appliance_')]

        cols_to_remove = []
        for col in appliance_cols_in_house:
            if col not in appliance_cols_in_training:
                cols_to_remove.append(col)

        if cols_to_remove:
            print(f"Removing missing appliance columns: {cols_to_remove}")
            house_df = house_df.drop(columns=cols_to_remove)
        # Ensure calculated columns are present
        if 'Rent Estimate to actual Price' not in house_df.columns:
            house_df['Rent Estimate to actual Price'] = 1.0
        if 'price to area average by zipcode' not in house_df.columns:
            house_df['price to area average by zipcode'] = 100.0
        if 'month listed' not in house_df.columns:
            house_df['month listed'] = 1

        # Set values for the appliance columns based on the house's appliances
        for col in all_appliances_cols:
            appliance_name = col.replace('Appliance_', '')
            house_df[col] = 1 if appliance_name in appliances_list else house_df[col] # Keep 0 if not in list

        # Drop the original 'Appliances' column if it exists after encoding
        if 'Appliances' in house_df.columns:
            house_df = house_df.drop(columns=['Appliances'])


        # Apply one-hot encoding to other string columns
        categorical_cols_to_encode = ['Cooling', 'Heating', 'Parking', 'Laundry', 'Home Type', 'Street Address', 'City', 'State', 'Zipcode'] # Added string columns that were not dropped
        categorical_cols_exist_in_house = [col for col in categorical_cols_to_encode if col in house_df.columns]
        house_df = pd.get_dummies(house_df, columns=categorical_cols_exist_in_house, dummy_na=False)


        # --- Prepare data for Regression Model ---
        # Ensure the input DataFrame has the same columns as the training data for the regression model
        training_columns_regression = X_regression.columns

        # Remove columns from input that training data doesn't have
        extra_cols = set(house_df.columns) - set(training_columns_regression)
        if 'Zipcode' in extra_cols:
            house_df = house_df.drop(columns=['Zipcode'])  # Training data didn't have Zipcode

        # Add missing columns from training data (skip appliance columns, we'll handle them by removal)
        missing_cols_regression = set(training_columns_regression) - set(house_df.columns)
        for c in missing_cols_regression:
            if not c.startswith('Appliance_'):  # Skip missing appliance columns
                house_df[c] = 0  # Other numeric columns default to 0

        # Ensure the order of columns is the same as training data
        house_df_regression = house_df[training_columns_regression].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Critical: Ensure the final dataframe columns match model feature names exactly
        model_features = regression_model.get_booster().feature_names

        # Check for missing columns in input data
        missing_in_input = set(model_features) - set(house_df_regression.columns)
        if missing_in_input:
            print(f"Warning: Input data missing {len(missing_in_input)} regression features")
            for col in list(missing_in_input)[:3]:  # Show first 3 missing
                print(f"  Missing: {col}")
            # Don't proceed with reordering if we're missing features
            raise ValueError(f"Regression model missing required features: {list(missing_in_input)[:5]}...")

        # Check for extra columns in input data
        extra_in_input = set(house_df_regression.columns) - set(model_features)
        if extra_in_input:
            print(f"Removing {len(extra_in_input)} extra columns from regression input data")
            house_df_regression = house_df_regression.drop(columns=list(extra_in_input))

        # Now reorder to match model expectations
        if list(house_df_regression.columns) != list(model_features):
            print(f"Reordering {len(model_features)} regression features to match model...")
            try:
                house_df_regression = house_df_regression[model_features]
                print("Regression feature reordering completed successfully")
            except Exception as e:
                print(f"Error during regression feature reordering: {e}")
                raise
        else:
            print("Regression features already in correct order")

        # Handle numerical columns for regression: fill NaNs with median from regression training and convert boolean to int
        for col in house_df_regression.columns:
            if pd.api.types.is_numeric_dtype(house_df_regression[col]):
                # Safe NaN filling with median from training data
                if col in X_regression.columns: # Check if the column was in the original regression training data
                    col_data = X_regression[col].dropna()
                    if not col_data.empty:
                        median_val = col_data.median()
                        house_df_regression[col] = house_df_regression[col].fillna(median_val)
                else: # If not in training data, fill with median of the current column
                    col_data = house_df_regression[col].dropna()
                    if not col_data.empty:
                        house_df_regression[col] = house_df_regression[col].fillna(col_data.median())
            elif pd.api.types.is_bool_dtype(house_df_regression[col]):
                house_df_regression[col] = house_df_regression[col].astype(int) # Convert boolean to int
            # Convert object columns that should be numeric
            elif pd.api.types.is_object_dtype(house_df_regression[col]):
                try:
                    house_df_regression[col] = pd.to_numeric(house_df_regression[col])
                    # Fill NaNs after conversion if any
                    if col in X_regression.columns:
                        median_val = X_regression[col].median()
                        house_df_regression[col] = house_df_regression[col].fillna(median_val)
                    else:
                        house_df_regression[col] = house_df_regression[col].fillna(house_df_regression[col].median())
                except ValueError:
                    # If conversion fails, drop the column as it's not numeric
                    print(f"Warning: Dropping non-numeric object column '{col}' from regression features.")
                    house_df_regression = house_df_regression.drop(columns=[col])


        house_df_classification = None
        classification_features = None
        if classification_model is not None:
            # --- Prepare data for Classification Model ---
            # Ensure the input DataFrame has the same columns as the training data for the classification model
            training_columns_classification = X_classification.columns

            # Remove columns from input that training data doesn't have (if any different from regression)
            extra_cols_classification = set(house_df.columns) - set(training_columns_classification)
            for c in extra_cols_classification:
                if c in house_df.columns:
                    house_df = house_df.drop(columns=[c])

            # Add missing columns from training data (skip missing appliance columns)
            missing_cols_classification = set(training_columns_classification) - set(house_df.columns)
            for c in missing_cols_classification:
                if not c.startswith('Appliance_'):  # Skip missing appliance columns
                    house_df[c] = 0  # Other numeric columns default to 0

            # Ensure the order of columns is the same as training data
            house_df_classification = house_df[training_columns_classification].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Critical: Ensure the final dataframe columns match model feature names exactly
            classification_features = classification_model.get_booster().feature_names

            # Check for missing columns in input data
            missing_in_input = set(classification_features) - set(house_df_classification.columns)
            if missing_in_input:
                print(f"Warning: Input data missing {len(missing_in_input)} classification features")
                for col in list(missing_in_input)[:3]:  # Show first 3 missing
                    print(f"  Missing: {col}")
                # Don't proceed with reordering if we're missing features
                raise ValueError(f"Classification model missing required features: {list(missing_in_input)[:5]}...")

            # Check for extra columns in input data
            extra_in_input = set(house_df_classification.columns) - set(classification_features)
            if extra_in_input:
                print(f"Removing {len(extra_in_input)} extra columns from input data")
                house_df_classification = house_df_classification.drop(columns=list(extra_in_input))

            # Now reorder to match model expectations
            if list(house_df_classification.columns) != list(classification_features):
                print(f"Reordering {len(classification_features)} classification features to match model...")
                try:
                    house_df_classification = house_df_classification[classification_features]
                    print("Classification feature reordering completed successfully")
                except Exception as e:
                    print(f"Error during classification feature reordering: {e}")
                    raise
            else:
                print("Classification features already in correct order")

            # Handle numerical columns for classification: fill NaNs with median from classification training and convert boolean to int
            for col in house_df_classification.columns:
                if pd.api.types.is_numeric_dtype(house_df_classification[col]):
                    # Safe NaN filling for classification
                    if col in X_classification.columns: # Check if the column was in the original classification training data
                        col_data = X_classification[col].dropna()
                        if not col_data.empty:
                            median_val = col_data.median()
                            house_df_classification[col] = house_df_classification[col].fillna(median_val)
                    else: # If not in training data, fill with median of the current column
                        col_data = house_df_classification[col].dropna()
                        if not col_data.empty:
                            house_df_classification[col] = house_df_classification[col].fillna(col_data.median())
                elif pd.api.types.is_bool_dtype(house_df_classification[col]):
                    house_df_classification[col] = house_df_classification[col].astype(int) # Convert boolean to int
                # Convert object columns that should be numeric
                elif pd.api.types.is_object_dtype(house_df_classification[col]):
                    try:
                        house_df_classification[col] = pd.to_numeric(house_df_classification[col])
                        house_df_classification[col] = house_df_classification[col].fillna(house_df_classification[col].median())
                    except ValueError:
                        print(f"Warning: Dropping non-numeric object column '{col}' from classification features.")
                        house_df_classification = house_df_classification.drop(columns=[col])


        # Verify regression model is loaded (classification optional)
        if regression_model is None:
            raise ValueError("Days-on-market regression model not loaded")
        if classification_model is None:
            print("Classification model not loaded - skipping hot/cold market classification")

        # Verify models have get_booster method (XGBoost models)
        if not hasattr(regression_model, 'get_booster'):
            raise ValueError("Regression model is not a valid XGBoost model")
        if classification_model is not None and not hasattr(classification_model, 'get_booster'):
            raise ValueError("Classification model is not a valid XGBoost model")

        # Feature validation - ensure columns match exactly
        if list(house_df_regression.columns) != list(regression_model.get_booster().feature_names):
            print(f"Warning: Feature mismatch in regression model - using reindexed data")

        # Predict using the regression model
        print("Starting regression prediction...")
        predicted_time_on_market = regression_model.predict(house_df_regression)[0]
        print(f"Regression prediction completed: {predicted_time_on_market}")
        # Predict using the classification model (if available)
        predicted_fast_sale = None
        predicted_fast_sale_proba = None
        if classification_model is not None:
            # Predict using the classification model
            predicted_fast_sale_proba = classification_model.predict_proba(house_df_classification)[:, 1][0]
            predicted_fast_sale = int(predicted_fast_sale_proba > 0.5)  # Assuming 0.5 as threshold for binary prediction
            print(f"Classification prediction completed: {predicted_fast_sale} (prob: {predicted_fast_sale_proba:.3f})")
        else:
            print("Classification model not loaded - skipping is_fast_sale prediction")




        return {
            'predicted_is_fast_sale': predicted_fast_sale,
            'predicted_time_on_market': predicted_time_on_market,
            'predicted_fast_sale_probability': predicted_fast_sale_proba, # Also return probability for insight
        }
    except Exception as e:
        print(f"Error in days_on_market prediction: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise

# Select a random row from df_shuffled and convert it to a dictionary
def generate_days_on_market_data(inputData, xgbr, xgbc, houseDict):
    # Convert "N/A" strings and other null representations to np.nan
    sample_house_row = {}
    for key, value in houseDict.items():
        if isinstance(value, str) and value.upper() in ['N/A', 'NA', 'NAN', 'NULL', '']:
            sample_house_row[key] = np.nan
        else:
            sample_house_row[key] = value

    # Apply comprehensive encoding to match the days-on-market model expectations
    sample_house_row = apply_comprehensive_encoding_for_days(sample_house_row, inputData)

    # Make predictions using the function and the sample row
    # Assuming 'xgbr' is the trained regression model and 'xgbc' is the trained classification model from previous steps

    predictions = predict_house_outcomes(sample_house_row, xgbr, xgbc, inputData)

    return predictions

def apply_comprehensive_encoding_for_days(house_row, inputData):
    """Apply comprehensive encoding to match the days-on-market model expectations"""

    # Calculate derived features if we have the necessary data
    price = house_row.get('Price', np.nan)
    rent_estimate = house_row.get('Rent Zestimate', np.nan)
    living_area = house_row.get('Living Area', np.nan)
    zipcode = house_row.get('Zipcode')

    # Add calculated features that the model expects
    if pd.notna(price) and pd.notna(rent_estimate) and rent_estimate != 0:
        house_row['Rent Estimate to actual Price'] = price / rent_estimate
    else:
        house_row['Rent Estimate to actual Price'] = np.nan

    # Calculate price per area ratios (simplified)
    if pd.notna(price) and pd.notna(living_area) and living_area != 0:
        house_row['price to area average by zipcode'] = price / living_area
    else:
        house_row['price to area average by zipcode'] = np.nan

    # Add month information (simplified - using current month, can be made more sophisticated)
    from datetime import datetime
    house_row['month listed'] = datetime.now().month

    # Remove Zipcode if present (not expected by days model)
    if 'Zipcode' in house_row:
        house_row.pop('Zipcode', None)

    # Initialize all expected appliance columns with 0
    expected_appliance_cols = [
        'Appliance_Bar Fridge', 'Appliance_Built In Microwave', 'Appliance_Built-In Oven',
        'Appliance_Convection Oven', 'Appliance_Cooktop', 'Appliance_Dishwasher', 'Appliance_Disposal',
        'Appliance_Double Oven', 'Appliance_Dryer', 'Appliance_ENERGY STAR Qualified Dishwasher',
        'Appliance_Electric Cooktop', 'Appliance_Electric Oven', 'Appliance_Electric Range',
        'Appliance_Electric Water Heater', 'Appliance_Exhaust Fan', 'Appliance_Freezer',
        'Appliance_Garbage Disposal', 'Appliance_Garbage disposal', 'Appliance_Gas Cooktop',
        'Appliance_Gas Oven', 'Appliance_Gas Range', 'Appliance_Gas Water Heater',
        'Appliance_Grill - Gas', 'Appliance_Grill - Other', 'Appliance_Ice Maker', 'Appliance_Instant Hot Water Gas',
        'Appliance_Kitchen Reverse Osmosis System', 'Appliance_Microwave', 'Appliance_None',
        'Appliance_Other', 'Appliance_Oven', 'Appliance_Oven/Range (Combo)', 'Appliance_Range',
        'Appliance_Range / Oven', 'Appliance_Range Hood', 'Appliance_Range Oven', 'Appliance_Range/Oven',
        'Appliance_Refrigerator', 'Appliance_Refrigerator W/IceMk', 'Appliance_Refrigerator-Ice maker',
        'Appliance_Refrigerator/Freezer', 'Appliance_Refrigerator/Icemaker', 'Appliance_RefrigeratorWithIce Maker',
        'Appliance_Self Cleaning Oven', 'Appliance_Smooth Stovetop Rnge', 'Appliance_Solar Hot Water Rented',
        'Appliance_Some Electric Appliances', 'Appliance_Stainless Steel Appliance(s)', 'Appliance_Stainless Steel Appliances',
        'Appliance_Stove', 'Appliance_Tankless Water Heater', 'Appliance_Trash Compactor', 'Appliance_WD Hookup',
        'Appliance_Wall Air Conditioning', 'Appliance_Wall Oven', 'Appliance_Warranty Provided', 'Appliance_Washer',
        'Appliance_Washer/Dryer', 'Appliance_Washer/Dryer Stacked', 'Appliance_Water Heater',
        'Appliance_Water Heater Leased', 'Appliance_Water Purifier', 'Appliance_Water Softener',
        'Appliance_Water Softener Owned', 'Appliance_Water Softener Rented', 'Appliance_Wine Cooler', 'Appliance_Wine Refrigerator'
    ]

    # Initialize all appliance columns to 0
    for col in expected_appliance_cols:
        house_row[col] = 0

    # Process appliances if present
    appliances_str = house_row.get('Appliances', '')
    if pd.notna(appliances_str) and isinstance(appliances_str, str):
        appliances_list = [app.strip() for app in str(appliances_str).split(',')]
        for appliance in appliances_list:
            appliance_col = f'Appliance_{appliance}'
            if appliance_col in house_row:
                house_row[appliance_col] = 1

    # Remove original Appliances column after encoding
    if 'Appliances' in house_row:
        house_row.pop('Appliances', None)

    return house_row