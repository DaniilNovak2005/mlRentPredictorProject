import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split # Although not used in this cell, include for completeness if needed elsewhere

# Assuming df and xgbr are already loaded and trained as per previous steps.
# If not, you would need to add code here to load the data and train the model.

def predict_new_house_price_combined(df, xgbr_model, new_house_data):
    print(new_house_data)
    """
    Predicts the price of a new house using a trained XGBoost model,
    handling preprocessing internally.

    Args:
        df (pd.DataFrame): The original DataFrame with house data.
        xgbr_model: The trained XGBoost Regressor model.
        new_house_data (dict): A dictionary containing the features of the new house.

    Returns:
        float: The predicted price of the new house.
    """
    # Convert "N/A" strings to NaN values
    for key, value in new_house_data.items():
        if value == 'N/A':
            new_house_data[key] = np.nan

    # Load and filter the data (replicated for self-containment in this function)
    # In a real scenario, you might pass the already filtered/processed df
    price_99th_percentile = df['Price'].quantile(0.99)
    df_filtered = df[df['Price'] <= price_99th_percentile].copy()


    # --- Preprocessing for finding the closest house ---
    # Apply one-hot encoding to 'Appliances' for the filtered data
    def encode_appliances_for_filtered(df_to_encode):
        appliances_list_series = df_to_encode['Appliances'].str.split(', ').apply(lambda x: x if isinstance(x, list) else [])
        all_appliances = sorted(list(set([item for sublist in appliances_list_series for item in sublist])))
        for appliance in all_appliances:
            df_to_encode[f'Appliance_{appliance}'] = appliances_list_series.apply(lambda x: 1 if appliance in x else 0)
        return df_to_encode

    df_filtered = encode_appliances_for_filtered(df_filtered)

    # Apply one-hot encoding to other string columns for the filtered data
    categorical_cols_df = ['Cooling', 'Heating', 'Parking', 'Laundry', 'Home Type']
    categorical_cols_exist = [col for col in categorical_cols_df if col in df_filtered.columns]
    df_filtered = pd.get_dummies(df_filtered, columns=categorical_cols_exist)

    # Identify numerical features for proximity calculation (excluding irrelevant)
    # Select only numerical columns after one-hot encoding
    numerical_features = df_filtered.select_dtypes(include=np.number).columns.tolist()
    irrelevant_columns = ['Price']
    numerical_features = [col for col in numerical_features if col not in irrelevant_columns]

    # Create a Series from the new house dictionary for numerical features
    new_house_numerical_for_distance = {col: new_house_data.get(col, np.nan) for col in numerical_features}
    new_house_numerical_series = pd.Series(new_house_numerical_for_distance)

    # --- Data Cleaning ---
    # Force all numerical columns to be numeric, coercing errors to NaN
    for col in numerical_features:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        new_house_numerical_series[col] = pd.to_numeric(new_house_numerical_series[col], errors='coerce')

    # --- Impute Missing Values using Geographical Proximity ---

    # Calculate distance based on Latitude and Longitude only
    coords = ['Latitude', 'Longitude']
    
    # Ensure the coordinate columns are numeric in both the dataframe and the new data
    for col in coords:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        new_house_data[col] = pd.to_numeric(new_house_data.get(col), errors='coerce')

    # Drop rows with missing coordinates from the search dataframe
    df_coords = df_filtered[coords].dropna()

    if pd.isna(new_house_data['Latitude']) or pd.isna(new_house_data['Longitude']):
        raise ValueError("Latitude or Longitude is missing for the new house, cannot impute values.")

    new_house_coords = pd.Series([new_house_data['Latitude'], new_house_data['Longitude']], index=coords)

    # Calculate Euclidean distance
    distances = np.sqrt(np.sum((df_coords - new_house_coords)**2, axis=1))
    
    # Get indices of all houses, sorted by distance
    sorted_indices = distances.sort_values().index

    # Loop through closest houses until all missing values are filled
    for index in sorted_indices:
        # Check if there are any missing values left to fill
        if not any(pd.isna(v) for v in new_house_data.values()):
            break
        
        closest_house_row = df_filtered.loc[index]
        
        # Iterate through the new house data and fill missing values
        for key, value in new_house_data.items():
            if pd.isna(value) or value is None:
                if key in closest_house_row and pd.notna(closest_house_row[key]):
                    new_house_data[key] = closest_house_row[key]
                    print(f"Filled missing value for '{key}' with: {closest_house_row[key]}")


    # --- Preprocessing for prediction ---
    # Convert the new house dictionary to a DataFrame
    new_house_df = pd.DataFrame([new_house_data])

    # Apply one-hot encoding to the same categorical columns as the training data
    # Use the original categorical column names from the new_house_data for get_dummies
    categorical_cols_new_house = [col for col in categorical_cols_df if col in new_house_df.columns]
    new_house_df = pd.get_dummies(new_house_df, columns=categorical_cols_new_house)

    # --- Define Training Columns ---
    # To get the correct training columns, we need to preprocess a copy of the original df
    # in the same way we will preprocess the new_house_df
    df_for_cols = df.copy()
    df_for_cols = encode_appliances_for_filtered(df_for_cols)
    df_for_cols = pd.get_dummies(df_for_cols, columns=categorical_cols_exist)
    
    # These are the columns that should not be considered features
    # These are the columns that should not be considered features, converted to lowercase for case-insensitive comparison
    non_feature_cols = [col.lower() for col in ['Street Address', 'City', 'State', 'URL', 'Price', 'Appliances', 'Cooling', 'Heating', 'Parking', 'Laundry', 'Home Type']]
    
    # Get the final list of training columns by comparing lowercase column names
    training_columns = [col for col in df_for_cols.columns if col.lower() not in non_feature_cols]


    all_appliances_cols_X = [col for col in training_columns if col.startswith('Appliance_')]

    # Apply one-hot encoding to 'Appliances' for the new house DataFrame based on training data's appliance columns
    if 'Appliances' in new_house_df.columns and isinstance(new_house_df['Appliances'].iloc[0], str):
        appliances_list = [appliance.strip() for appliance in new_house_df['Appliances'].iloc[0].split(',')]
    else:
        appliances_list = []

    # Ensure all appliance columns from training are in new_house_df before setting values
    for col in all_appliances_cols_X:
        if col not in new_house_df.columns:
            new_house_df[col] = 0 # Initialize with 0

    for col in all_appliances_cols_X:
        appliance_name = col.replace('Appliance_', '')
        new_house_df[col] = 1 if appliance_name in appliances_list else new_house_df[col] # Keep 0 if not in list


    # Drop the original 'Appliances' column if it exists after encoding
    if 'Appliances' in new_house_df.columns:
        new_house_df = new_house_df.drop(columns=['Appliances'])


    # Align columns with the training data (X)
    missing_cols = set(training_columns) - set(new_house_df.columns)
    for c in missing_cols:
        new_house_df[c] = 0

    # Ensure the order of columns is the same
    new_house_df = new_house_df[training_columns]
    new_house_df['Zipcode'] = int(new_house_df['Zipcode'])
    new_house_df = new_house_df.drop(columns=["Zipcode"])
    # Predict the price
    dmatrix = xgb.DMatrix(new_house_df)
    predicted_price = xgbr_model.predict(dmatrix)

    return int(predicted_price[0])

# Define the new house dictionary for testing
new_house_for_prediction = {
    'Bedrooms': 6.0,
    'Bathrooms': 7.0,
    'Living Area': 3500.0,
    'Year Built': 2000.0,
    'Property Tax': 2500.0,
    'Property Tax Rate': 1.5,
    'Latitude': 28.0,
    'Longitude': -81.0,
    'Last Rental Price': 9800.0,
    'Times Rented': 5,
    'Years Since Rented': 1.0,
    'CAGR': 0.05,
    'Avg Annual Value Increase': 1000.0,
    'Avg Annual Tax Increase': 50.0,
    'Std Dev Value Increase': 200.0,
    'Std Dev Tax Increase': 10.0,
    'Walk Score': 50.0,
    'Transit Score': 30.0,
    'Bike Score': 70.0,
    'Primary School Distance': 1.0,
    'Primary School Rating': 7.0,
    'Middle School Distance': 1.5,
    'Middle School Rating': 6.0,
    'High School Distance': 2.0,
    'High School Rating': 7.0,
    'Appliances': 'Dishwasher, Dryer, Refrigerator, Washer',
    'Cooling': 'Central Air',
    'Heating': 'Central, Electric',
    'Laundry': 'In Unit',
    'Parking': 'Garage',
    'Home Type': 'SINGLE_FAMILY',
    'zipcode': 33605.0,
    'median_household_income': 120000.0,
    'average_household_income': 70000.0,
    'per_capita_income': 30000.0,
    'high_income_households_percent': 5.0,
    'median_income_25_to_44_years': 65000.0,
    'median_income_45_to_64_years': 75000.0,
    'median_income_65_years_and_over': 40000.0,
    'median_income_under_25_years': 35000.0,
    'Street Address': '123 Main St',
    'City': 'Tampa',
    'State': 'FL',
    'URL': 'http://example.com'
}


# Example usage:
# Assuming df and xgbr are already loaded and trained as per previous steps
# predicted_price = predict_new_house_price_combined(df.copy(), xgbr, new_house_for_prediction)

# if predicted_price is not None:
#     print(f"Predicted price for the new house: {predicted_price}")