import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

# Standardize the features
scaler = StandardScaler()

# Load the dataset (only first 100 rows)
file_path = "merged_output_fixed (1).csv"  # Replace with your file path if different
data = pd.read_csv(file_path)

# Features and target
X = data[["size","height","no_tasks","kitchen","wardrobe","bath","doors","windows","interior_walls","floor_area"]]
y = data['module_time']

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

# Set up k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(mean_squared_error, greater_is_better=False)  # Negative MSE for scoring compatibility

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate models
results = {}
feature_importances = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test data
    y_pred = model.predict(X_test)
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importances[name] = model.coef_

# Train and evaluate models using cross-validation
for name, model in models.items():
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)
    # Convert negative MSE back to positive
    mse_scores = -cv_scores
    # Store mean and standard deviation of MSE
    results[name] = {"Mean MSE": mse_scores.mean(), "Std MSE": mse_scores.std(), "r2": r2_score(y_test, y_pred)}

# Print cross-validation results
print("Cross-Validation Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: Mean MSE = {metrics['Mean MSE']:.2f}, Std MSE = {metrics['Std MSE']:.2f}, r2 = {metrics['r2']:.2f}")


    # Save the best performing model

    # Find the best model based on Mean MSE
best_model_name = min(results, key=lambda x: results[x]['Mean MSE'])
best_model = models[best_model_name]

# Save the model to a file
model_filename = f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
joblib.dump(best_model, model_filename)

print(f"Best model ({best_model_name}) saved as {model_filename}")
# Plot covariance matrix for each model
for name, model in models.items():
    # Predict on test data
    y_pred = model.predict(X_test)
    # Calculate residuals
    residuals = y_test - y_pred
    # Create a DataFrame with features and residuals
    df_residuals = pd.DataFrame(X_test, columns=X.columns)
    df_residuals['residuals'] = residuals
    # Calculate covariance matrix
    covariance_matrix = df_residuals.cov()
    # Plot covariance matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f"Covariance Matrix for {name}")
    plt.show()

    # Plot feature importances for each model   
    for name, importances in feature_importances.items():
        plt.figure(figsize=(10, 6))
        if name == "Linear Regression":
            sns.barplot(x=importances, y=X.columns)
            plt.title(f"Correlation Coefficients for {name}")
        else:
            sns.barplot(x=importances, y=X.columns)
            plt.title(f"Feature Importances for {name}")
        plt.show()
