import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_original_data(data_dir):
    """Load the original data"""
    input_file = os.path.join(data_dir, 'Datasets', 'soildataset.xlsx')
    data = pd.read_excel(input_file)
    return data

def extract_water_content(record):
    """Extract water content from record name"""
    try:
        water_ml = int(str(record).split('_')[1].split('ml')[0])
        return water_ml
    except:
        return None

def plot_spectral_curves_by_water(data, output_dir):
    """Plot spectral reflectance curves grouped by water content"""
    # Extract water content from Records column
    data['water_content'] = data['Records'].apply(extract_water_content)
    
    # Get wavelength columns (columns 2-19)
    wavelength_cols = data.columns[1:19].tolist()
    wavelengths = [410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940]
    
    # Target columns (soil properties)
    target_cols = [
        'Capacitity Moist', 'Temp', 'Moist', 'EC (u/10 gram)', 'Ph',
        'Nitro (mg/10 g)', 'Posh Nitro (mg/10 g)', 'Pota Nitro (mg/10 g)'
    ]
    
    # Water content levels
    water_levels = [0, 25, 50, 75, 100]
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    # Colors for different water contents
    colors = ['#FF9900', '#FF5733', '#C70039', '#900C3F', '#581845']
    
    # Plot for each target property
    for target_col in target_cols:
        plt.figure(figsize=(12, 8))
        
        # Plot curves for each water content level
        for i, water_level in enumerate(water_levels):
            # Get samples with this water content
            samples = data[data['water_content'] == water_level].head(10)
            
            if len(samples) > 0:
                # Calculate mean spectral values for this water content
                mean_spectral = []
                for col in wavelength_cols:
                    mean_spectral.append(samples[col].mean())
                
                # Plot the curve
                plt.plot(wavelengths, mean_spectral, '-o', color=colors[i], 
                        markersize=6, label=f'{water_level}ml water content')
        
        plt.title(f'Spectral Reflectance Curves for {target_col}\nGrouped by Water Content')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Set axis properties
        plt.xticks(wavelengths, rotation=45)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'water_content_spectra_{target_col.replace("/", "_")}.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

        # Now create individual plots for each water content level
        for water_level in water_levels:
            plt.figure(figsize=(12, 8))
            
            # Get samples with this water content
            samples = data[data['water_content'] == water_level].head(10)
            
            if len(samples) > 0:
                # Plot individual curves for each sample
                for idx, row in samples.iterrows():
                    spectral_values = [row[col] for col in wavelength_cols]
                    plt.plot(wavelengths, spectral_values, '-o', markersize=4,
                            label=f'Sample {row["Records"]}')
            
            plt.title(f'Spectral Reflectance Curves for {target_col}\n{water_level}ml Water Content')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Set axis properties
            plt.xticks(wavelengths, rotation=45)
            
            # Add legend
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'spectra_{water_level}ml_{target_col.replace("/", "_")}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'analysis')
    
    # Load original data
    print("Loading original data...")
    data = load_original_data(current_dir)
    
    # Create spectral plots grouped by water content
    print("\nCreating spectral reflectance plots grouped by water content...")
    plot_spectral_curves_by_water(data, output_dir)
    
    print(f"\nAnalysis complete! Visualizations saved to: {output_dir}") 