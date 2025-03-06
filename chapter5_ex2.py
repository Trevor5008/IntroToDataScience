from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

feature_names = diabetes.feature_names
# targetNames = diabetes.target_names

print(feature_names)
print(y)
# 2. Split into training and testing sets (lets do 75% train, 25% test for variety)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)  # finds the best-fitting linear relationship

# 4. Predict on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test set:", mse)

# Scatter plot of Actual vs Predicted values
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Disease Progression')
plt.ylabel('Predicted Disease Progression')
plt.title('Actual vs Predicted Disease Progression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Ideal Prediction (Perfect accuracy)' )
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Disease Progression')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()