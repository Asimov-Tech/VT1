import pandas as pd
import joblib
import argparse

# Load the model
model = joblib.load('random_forest_model.pkl')

# Function to make predictions based on a CSV file
def predict_from_csv(csv_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    
    # Print the columns of the DataFrame to debug
    print("Columns in the CSV file:", data.columns.tolist())
    
    # Ensure the columns are in the correct order
    columns = ["size", "height", "no_tasks", "kitchen", "wardrobe", "bath", "doors", "windows", "interior_walls", "floor_area"]
    data = data[columns]
    
    # Make predictions
    predictions = model.predict(data)
    
    # Add predictions to the DataFrame
    data['predictions'] = predictions
    
    # Save the DataFrame with predictions to a new CSV file
    data.to_csv('predictions_output.csv', index=False)
    
    return predictions

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict from CSV file using a pre-trained model.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    args = parser.parse_args()
    
    predictions = predict_from_csv(args.csv_file)
    print(predictions)