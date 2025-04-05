import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def load_data():
    """Load the normalized features and targets"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features_file = os.path.join(current_dir, 'processed', 'Normalized_features.xlsx')
    targets_file = os.path.join(current_dir, 'processed', 'Normalized_targets.xlsx')
    
    print("Loading normalized data...")
    X = pd.read_excel(features_file)
    y = pd.read_excel(targets_file)
    
    return X, y

def select_features(X, y, property_name):
    """Select most important features using Random Forest"""
    print(f"\nSelecting features for {property_name}...")
    
    # Use Random Forest for feature selection
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = SelectFromModel(rf, prefit=False)
    selector.fit(X, y)
    
    # Get selected feature mask and names
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} features")
    
    return X[selected_features], selected_features

def optimize_model(X, y, property_name):
    """Find optimal hyperparameters using grid search"""
    print(f"\nOptimizing model for {property_name}...")
    
    # Define parameter grid (smaller for faster training)
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64)],
        'activation': ['relu'],
        'alpha': [0.001, 0.01],
        'learning_rate': ['adaptive']
    }
    
    # Create base model
    model = MLPRegressor(max_iter=500, early_stopping=True, random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='r2', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation R²: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def plot_feature_importance(X, y, property_name, output_dir):
    """Plot feature importance using Random Forest"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['importance'])
    plt.title(f'Feature Importance for {property_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_{property_name.replace("/", "_")}.png'))
    plt.close()

def plot_predictions(y_true, y_pred, property_name, output_dir):
    """Plot predictions vs actual values"""
    # Convert predictions to absolute values
    y_pred_abs = np.abs(y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred_abs, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', lw=2)
    
    # Calculate R² using absolute values
    r2 = r2_score(y_true, y_pred_abs)
    mse = mean_squared_error(y_true, y_pred_abs)
    
    plt.title(f'{property_name} - Predictions vs Actual\nR² = {r2:.3f}, MSE = {mse:.3f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values (Absolute)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'predictions_{property_name.replace("/", "_")}.png'))
    plt.close()
    
    # Print conversion statistics
    neg_count = (y_pred < 0).sum()
    if neg_count > 0:
        print(f"Converted {neg_count} negative predictions to positive for {property_name}")
    
    return r2, mse

def save_model(model, selected_features, property_name, output_dir):
    """Save the trained model and selected features"""
    # Save model
    model_path = os.path.join(output_dir, f'soil_property_model_{property_name.replace("/", "_")}.joblib')
    joblib.dump(model, model_path)
    
    # Save selected features
    features_path = os.path.join(output_dir, f'selected_features_{property_name.replace("/", "_")}.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(selected_features))
    
    print(f"Model and features saved for {property_name}")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory for model artifacts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X, y = load_data()
    
    # Train separate model for each property
    results = {}
    for property_name in y.columns:
        print(f"\n{'='*50}")
        print(f"Training model for {property_name}")
        print('='*50)
        
        # Select features
        X_selected, selected_features = select_features(X, y[property_name], property_name)
        
        # Plot feature importance
        plot_feature_importance(X_selected, y[property_name], property_name, output_dir)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y[property_name], test_size=0.2, random_state=42
        )
        
        # Optimize and train model
        model = optimize_model(X_train, y_train, property_name)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot and evaluate using absolute values
        r2, mse = plot_predictions(y_test, y_pred, property_name, output_dir)
        
        # Save results
        results[property_name] = {
            'MSE': mse,
            'R2': r2,
            'Features': len(selected_features)
        }
        
        # Save model and features
        save_model(model, selected_features, property_name, output_dir)
    
    # Save overall results
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, 'model_metrics.csv'))
    
    # Print final results
    print("\nFinal Model Performance (using absolute values):")
    for property_name, metrics in results.items():
        print(f"\n{property_name}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  Selected Features: {metrics['Features']}")
    
    print("\nTraining complete! Models and evaluation results saved to:", output_dir)

if __name__ == "__main__":
    main() 