import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def load_models_and_features(models_dir):
    """Load all trained models and their selected features"""
    models = {}
    features = {}
    
    # Get all model files
    model_files = [f for f in os.listdir(models_dir) if f.startswith('soil_property_model_')]
    
    for model_file in model_files:
        # Extract property name from filename
        property_name = model_file.replace('soil_property_model_', '').replace('.joblib', '')
        
        # Load model
        model_path = os.path.join(models_dir, model_file)
        models[property_name] = joblib.load(model_path)
        
        # Load selected features
        features_file = f'selected_features_{property_name}.txt'
        features_path = os.path.join(models_dir, features_file)
        with open(features_path, 'r') as f:
            features[property_name] = f.read().splitlines()
    
    return models, features

def plot_spectral_curves(spectral_data, predictions, output_dir):
    """
    Create line charts showing wavelength vs intensity for each soil property.
    
    Args:
        spectral_data (pd.DataFrame): DataFrame containing spectral measurements
        predictions (pd.DataFrame): DataFrame containing predicted soil properties
        output_dir (str): Directory to save the plots
    """
    # Get wavelength columns (numeric columns from 410 to 940)
    wavelength_cols = [col for col in spectral_data.columns if str(col).isdigit()]
    wavelength_cols.sort(key=lambda x: int(x))
    wavelengths = [int(col) for col in wavelength_cols]
    
    # Create a plot for each property
    for property_name in predictions.columns:
        plt.figure(figsize=(10, 6))
        
        # Plot spectral curves for all samples
        for i in range(len(spectral_data)):
            intensities = spectral_data.iloc[i][wavelength_cols].values
            property_value = predictions.iloc[i][property_name]
            plt.plot(wavelengths, intensities, '-', linewidth=2, 
                    label=f'Sample {i+1} ({property_value:.2f})')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(f'Spectral Curves for {property_name}')
        plt.grid(True)
        
        # Add minor gridlines
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'spectral_curves_{property_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Created {len(predictions.columns)} spectral curve plots")

def predict_properties(models, features, spectral_data):
    """Predict soil properties from spectral data"""
    predictions = {}
    
    # Load original target data to get mean and std for denormalization
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_targets = pd.read_excel(os.path.join(current_dir, 'Datasets', 'soildataset.xlsx'))
    
    # Make predictions for each property
    for property_name in models.keys():
        # Select features for this property
        X = spectral_data[features[property_name]]
        
        # Make prediction
        pred = models[property_name].predict(X)
        
        # Denormalize predictions
        original_name = property_name.replace('_', '/')  # Convert back to original name format
        if original_name in original_targets.columns:
            mean = original_targets[original_name].mean()
            std = original_targets[original_name].std()
            pred = (pred * std) + mean
        
        # Ensure predictions are positive
        pred = np.abs(pred)
        predictions[property_name] = pred
        
        print(f"\n{property_name}:")
        print(f"  Range: {pred.min():.2f} to {pred.max():.2f}")
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def main():
    # Create predictions directory if it doesn't exist
    predictions_dir = os.path.join('src', 'data', 'models', 'predictions')
    plots_dir = os.path.join(predictions_dir, 'plots')
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("\nLoading trained models...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    models, features = load_models_and_features(models_dir)

    print("\nLoading example spectral data...")
    example_file = os.path.join(current_dir, 'processed', 'Normalized_features.xlsx')
    example_data = pd.read_excel(example_file).head(5)  # Using first 5 samples as example

    print("\nMaking predictions...")
    predictions = predict_properties(models, features, example_data)

    # Print ranges for each property
    for column in predictions.columns:
        print(f"\n{column}:")
        print(f"  Range: {predictions[column].min():.2f} to {predictions[column].max():.2f}")

    print("\nCreating spectral curves plots...")
    plot_spectral_curves(example_data, predictions, plots_dir)

    # Save predictions
    output_file = os.path.join(predictions_dir, 'soil_predictions.xlsx')
    try:
        predictions.to_excel(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving predictions: {str(e)}")
        print("Please ensure you have write permissions to the directory.")

    print("\nPlots saved to:", plots_dir)

if __name__ == "__main__":
    main() 