import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def preprocess_data(df, label_encoder=None):
    """
    Preprocess the input DataFrame.
    """
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['State/UnionTerritory'] = label_encoder.fit_transform(df['State/UnionTerritory'])
    else:
        df['State/UnionTerritory'] = label_encoder.transform(df['State/UnionTerritory'])
    df['Date'] = pd.to_datetime(df['Date']).astype(int) / 10**9  # Convert to Unix timestamp
    df['Time'] = pd.to_datetime(df['Time']).astype(int) / 10**9  # Convert to Unix timestamp
    return df, label_encoder

# Load and preprocess training data
df = pd.read_csv('/content/train_data_covid (1).csv')
df, label_encoder = preprocess_data(df)
X = df.drop(['Sno', 'Deaths', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis=1)
y = df['Deaths']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Normalize the data
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Define base models for stacking
base_models = [
    ('decision_tree', DecisionTreeRegressor(random_state=23)),
    ('gradient_boosting', GradientBoostingRegressor(random_state=23))
]

# Define the stacked model
stacked_model = StackingRegressor(estimators=base_models, final_estimator=GradientBoostingRegressor(random_state=23))

# Fit the stacked model
stacked_model.fit(X_train_scaled, y_train_scaled)

# Make predictions
predictions_scaled = stacked_model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted')
plt.title('Deaths Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.legend()
plt.show()

# Preprocess the test data for predictions
test_df = pd.read_csv("/content/test_data_covid (2).csv")
test_df['State/UnionTerritory'] = test_df['State/UnionTerritory'].replace({
    'Madhya Pradesh***': 'Madhya Pradesh',
    'Himanchal Pradesh': 'Himachal Pradesh',
    'Karanataka': 'Karnataka',
    'Maharashtra***': 'Maharashtra',
    'Dadra and Nagar Haveli': 'Goa',
    'Bihar****': 'Bihar'
})
test_df, _ = preprocess_data(test_df, label_encoder)
X_test_final = test_df.drop(['Sno', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis=1)
X_test_final_scaled = scaler_x.transform(X_test_final)

# Make final predictions
final_predictions_scaled = stacked_model.predict(X_test_final_scaled)
final_predictions = scaler_y.inverse_transform(final_predictions_scaled.reshape(-1, 1)).flatten()

# Round off the predictions
final_predictions = np.round(final_predictions)

# Save the rounded predictions to a CSV file
predictions_df = pd.DataFrame({
    'Sno': test_df['Sno'],
    'Predicted_Deaths': final_predictions
})
predictions_df.to_csv('/content/predicted_deaths_rounded.csv', index=False)
print("Rounded predictions saved to 'predicted_deaths_rounded.csv'")

# Download the CSV file if using Google Colab
from google.colab import files
files.download('/content/predicted_deaths_rounded.csv')
