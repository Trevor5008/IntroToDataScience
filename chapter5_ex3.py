from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd


def evaluatePredictions(y_test,y_pred):
    # Evaluate predictions
    metrics = []
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error (MSE)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Root Mean Squared Error (RMSE)
    nrmse = ((rmse) / (max(y_test) - min(y_test))) * 100  # Normalized Root Mean Squared Error (NRMSE)
    mae = mean_absolute_error(y_test, y_pred)  #
    r2 = r2_score(y_test, y_pred)

    # Store the results in lists
    metrics.append([mse, rmse, nrmse, mae, r2])
    return metrics

def perform_linearRegression():
    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    metrics = evaluatePredictions(y_test, y_pred)
    # Plot Actual vs Predicted values
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Prices (in $100,000s)")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted California Housing Prices")

    # Perfect predictions line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    # plt.show()

    return metrics,plt

def perform_randomForest():
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    metrics = evaluatePredictions(y_test, y_pred_rf)
    # Plot Actual vs Predicted values
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred_rf, alpha=0.6)
    plt.xlabel("Actual Prices (in $100,000s)")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted California Housing Prices")

    # Perfect predictions line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    return metrics,plt

if __name__ == '__main__':
    # Load California housing dataset
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    print(X)
    print(y)
    print(feature_names)
    df = pd.DataFrame(X, columns=housing.feature_names)
    df['Price'] = y
    print(df.head())

    # Split data (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    resultsLR = perform_linearRegression()
    print("Linear Regression Metrics:", resultsLR[0])
    resultsLR[1].show()

    resultsRF = perform_randomForest()
    print("Random Forest Metrics:", resultsRF[0])
    resultsRF[1].show()
