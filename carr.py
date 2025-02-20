# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Select relevant features and target variable
features = ['MPG.city', 'EngineSize', 'Horsepower', 'RPM', 'Weight', 'Length', 'Width']
target = 'Price'

# Handle missing values (if any)
data = data.dropna(subset=features + [target])

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualization: Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.show()

# Visualization: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='red')
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()

# Example prediction
example_car = np.array([[30, 2.0, 140, 5500, 3000, 180, 70]])
predicted_price = model.predict(example_car)
print(f'Predicted Price for the example car: {predicted_price[0]}')
