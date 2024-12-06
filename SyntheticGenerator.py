from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import math

# Import necessary libraries
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\silas\Desktop\Manufactoring Technology 1\Applied Machine Learning\output.csv")

# Select relevant columns
df = df[['module_time',  'size',  'height',  'no_tasks',  'kitchen',  'wardrobe',  'bath',  'doors',  'windows',  'interior_walls',  'floor_area']]

# Define means and standard deviations for each feature
size_means = [35,40,50]
size_std = 5

height_means = [3]
height_std = 0.5

no_tasks_means = [20,24,27]
no_tasks_std = 2

kitchen_means = [0,1]
kitchen_std = None

wardrobe_means = [3]
wardrobe_std = 1

bath_means = [0,1]
bath_std= None

doors_means = [3]
doors_std = 1

windows_means = [2,3,4]
windows_std = 1

interior_walls_means = [2,2,4]
interior_walls_std = 1

floor_area_means = [20,25,35]
floor_area_std = 2

# Function to generate a data point based on threshold and distribution parameters
def generate_point(threshold, means, std):
    if len(means) == 1:
        value = np.random.normal(means[0], std)
    if len(means) == 2:
        if threshold == 'low':
            value = 0
        if threshold == 'medium':
            value = np.random.randint(2, size=1)[0]
        if threshold == 'high':
            value = 1
    if len(means) == 3:
        if threshold == 'low':
            value = np.random.normal(means[0], std)
        if threshold == 'medium':
            value = np.random.normal(means[1], std)
        if threshold == 'high':
            value = np.random.normal(means[2], std)
    if value < 0:
        value = 0
    return value

# Function to generate a synthetic dataset
def generate_dataset(n_samples, thres_low, thres_high):
    synthetic_data = []
    np.random.seed(42)

    for _ in range(n_samples):
        module_time = np.random.randint(200, 600, size=1)[0]
        if module_time < thres_low:
            threshold = 'low'
        elif module_time > thres_high:
            threshold = 'high'
        else:
            threshold = 'medium'

        new_row = {
            'module_time': module_time,
            'size': int(generate_point(threshold, size_means, size_std)),
            'height': np.round(generate_point(threshold, height_means, height_std), 2),
            'no_tasks': int(generate_point(threshold, no_tasks_means, no_tasks_std)),
            'kitchen': int(generate_point(threshold, kitchen_means, kitchen_std)),
            'wardrobe': int(generate_point(threshold, wardrobe_means, wardrobe_std)),
            'bath': int(generate_point(threshold, bath_means, bath_std)),
            'doors': int(generate_point(threshold, doors_means, windows_std)),
            'windows': int(generate_point(threshold, windows_means, windows_std)),
            'interior_walls': int(generate_point(threshold, interior_walls_means, interior_walls_std)),
            'floor_area': int(generate_point(threshold, floor_area_means, floor_area_std))
        }
        synthetic_data.append(new_row)
    return pd.DataFrame(synthetic_data)

# Generate synthetic data
synthetic_data = generate_dataset(6000, 325, 475)

# Remove any outliers where the floor area is bigger than the module size
synthetic_data = synthetic_data[synthetic_data['size'] >= synthetic_data['floor_area']]

# Save the synthetic data to a CSV file
synthetic_data.to_csv('output_synthetic.csv', index=False)

# Calculate correlations with 'module_time'
correlations = synthetic_data.corr()['module_time'].drop(['module_time'])

# Print the correlations
print(correlations)

# Plot the correlations
plt.figure(figsize=(8, 6))
sns.barplot(x=correlations.index, y=correlations.values, palette='coolwarm')

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Correlation with module_time')
plt.title('Correlation between module_time and Other Features')
plt.xticks(rotation=45)
plt.show()