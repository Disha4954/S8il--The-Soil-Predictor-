import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
import traceback

def load_raw_data():
    """Load the raw soil dataset."""
    # Set the correct data path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Datasets', 'soildataset.xlsx')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_excel(data_path)
    return df

def extract_sample_info(sample_name):
    """Extract soil number, water content, and test number from sample name."""
    # Pattern: {soil_number}_{water_content}ml-{test_number}
    # Example: 168_50ml-6
    pattern = r'^(\d+)_(\d+)ml-(\d+)$'
    match = re.match(pattern, str(sample_name).strip())
    
    if match:
        soil_number = int(match.group(1))
        water_content = int(match.group(2))
        test_number = int(match.group(3))
        return soil_number, water_content, test_number
    else:
        raise ValueError(f"Invalid sample name format: '{sample_name}'. Expected format: number_waterml-number (e.g., 168_50ml-6)")

def separate_features_targets(df):
    """Separate features (columns 2-19) and target variables."""
    # Get all columns
    all_columns = df.columns.tolist()
    
    # Use column indices 2-19 for features (0-based index)
    feature_columns = all_columns[1:19]  # This will get columns 2-19
    target_columns = all_columns[19:]    # This will get all remaining columns
    
    features = df[feature_columns]
    targets = df[target_columns]
    
    # Extract sample information from the Records column
    sample_info = []
    for sample_name in df['Records']:  # Get sample names from Records column
        soil_number, water_content, test_number = extract_sample_info(str(sample_name))
        sample_info.append({
            'soil_number': soil_number,
            'water_content': water_content,
            'test_number': test_number
        })
    
    # Create DataFrame with sample information
    sample_info_df = pd.DataFrame(sample_info, index=df.index)
    
    # Combine sample information with features
    features = pd.concat([sample_info_df, features], axis=1)
    
    return features, targets

def normalize_data(features, targets):
    """Normalize the data using StandardScaler."""
    # Convert all column names to strings
    features.columns = features.columns.astype(str)
    targets.columns = targets.columns.astype(str)
    
    # Separate numerical and categorical columns
    numerical_columns = features.select_dtypes(include=[np.number]).columns
    categorical_columns = features.select_dtypes(include=['object']).columns
    
    # Normalize numerical features
    feature_scaler = StandardScaler()
    normalized_features = feature_scaler.fit_transform(features[numerical_columns])
    normalized_features_df = pd.DataFrame(normalized_features, 
                                        columns=numerical_columns,
                                        index=features.index)
    
    # Add back categorical columns if any
    if len(categorical_columns) > 0:
        normalized_features_df = pd.concat([normalized_features_df, features[categorical_columns]], axis=1)
    
    # Normalize targets
    target_scaler = StandardScaler()
    normalized_targets = target_scaler.fit_transform(targets)
    normalized_targets_df = pd.DataFrame(normalized_targets, 
                                       columns=targets.columns,
                                       index=targets.index)
    
    return normalized_features_df, normalized_targets_df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Fill missing values with mean for numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    
    # Fill missing values with mode for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    return df

def preprocess_data(input_file, output_dir):
    """
    Preprocess the soil data by separating features and targets,
    and normalizing them using StandardScaler.
    
    Args:
        input_file (str): Path to the input Excel file
        output_dir (str): Directory to save the processed data
    """
    try:
        # Read the input data
        print(f"Reading data from: {input_file}")
        data = pd.read_excel(input_file)
        
        # Print column names for debugging
        print("\nAvailable columns in the dataset:")
        print(data.columns.tolist())
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract water content from Records column
        water_content = []
        for record in data['Records']:
            try:
                # Extract water content from the record name (format: number_waterml-number)
                water_ml = int(str(record).split('_')[1].split('ml')[0])
                water_content.append(water_ml)
            except:
                print(f"Warning: Could not extract water content from record: {record}")
                water_content.append(0)
        
        # Get wavelength columns (columns 2-19)
        all_columns = data.columns.tolist()
        wavelength_cols = all_columns[1:19]  # Python uses 0-based indexing
        print("\nWavelength columns (columns 2-19):")
        print(wavelength_cols)
        
        # Create feature data with water_content and wavelengths
        feature_dict = {'water_content': water_content}
        
        # Add wavelength columns
        for wavelength in wavelength_cols:
            wavelength_data = pd.to_numeric(data[wavelength], errors='coerce')
            wavelength_data = wavelength_data.fillna(wavelength_data.mean())
            feature_dict[str(wavelength)] = wavelength_data
        
        # Create feature DataFrame all at once to avoid fragmentation
        feature_data = pd.DataFrame(feature_dict)
        
        # Create target data with correct column names
        target_columns = [
            'Capacitity Moist', 'Temp', 'Moist', 'EC (u/10 gram)', 'Ph',
            'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)'
        ]
        
        # Check which target columns exist in the dataset
        available_targets = [col for col in target_columns if col in data.columns]
        if not available_targets:
            raise ValueError("No target columns found in the dataset. Please check the column names.")
        
        # Convert target columns to numeric and handle missing values
        target_dict = {}
        for col in available_targets:
            target_values = pd.to_numeric(data[col], errors='coerce')
            target_values = target_values.fillna(target_values.mean())
            target_dict[col] = target_values
        
        # Create target DataFrame all at once to avoid fragmentation
        target_data = pd.DataFrame(target_dict)
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = pd.DataFrame(
            scaler.fit_transform(feature_data),
            columns=feature_data.columns
        )
        
        # Normalize targets
        target_scaler = StandardScaler()
        normalized_targets = pd.DataFrame(
            target_scaler.fit_transform(target_data),
            columns=target_data.columns
        )
        
        # Save processed data
        output_features = os.path.join(output_dir, 'Normalized_features.xlsx')
        output_targets = os.path.join(output_dir, 'Normalized_targets.xlsx')
        
        print(f"\nSaving normalized features to: {output_features}")
        normalized_features.to_excel(output_features, index=False)
        
        print(f"Saving normalized targets to: {output_targets}")
        normalized_targets.to_excel(output_targets, index=False)
        
        print(f"\nProcessed data saved to: {output_dir}")
        print("\nFeature data shape:", normalized_features.shape)
        print("Target data shape:", normalized_targets.shape)
        print("\nTarget columns used:", available_targets)
        print("\nWavelength columns used:", wavelength_cols)
        
        return normalized_features, normalized_targets
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set input and output paths
    input_file = os.path.join(current_dir, 'Datasets', 'soildataset.xlsx')
    output_dir = os.path.join(current_dir, 'processed')
    
    # Run preprocessing
    preprocess_data(input_file, output_dir) 