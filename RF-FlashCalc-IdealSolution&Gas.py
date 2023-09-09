# -*- coding: utf-8 -*-
"""
Random Forest for Flash calculation considering Ideal solution
The 12th International Chemical Engineering Congress & Exhibition (IChEC 2023)
@author: Shabnam Shahhoseyni/ PhD Candidate/ Amirkabir University of Technology
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel('flash_calculation_results-original.xlsx')

# Fill NaN values in mixture components (columns 5 to 15) with 0
data.iloc[:, 4:15] = data.iloc[:, 4:15].fillna(0)

# Drop rows with missing values in the 'Vapor Split (V/F)' column    
data.dropna(subset=['Vapor Split (V/F)'], inplace=True)

# Extract features (Temperature, Pressure, and Component Compositions) and targets (Vapor Compositions, Liquid Compositions)
X = data[['Temperature (K)', 'Pressure (bar)'] + list(data.columns[4:15])].values
y = data[[ 'Bubble Pres. (bar)', 'Dew Pres. (bar)', 'Bubble Temp (K)', 'Dew Point Temp. (K)','Vapor Split (V/F)']].values

# Normalize the inputs and outputs
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42)



# Create a Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_train = rf_model.predict(X_train)


# Calculate MSE and R2 for all outputs
mse_train = mean_squared_error(y_train, y_pred_train, multioutput='raw_values')
print("\nMSE-train:", mse_train)
R2_score_train = r2_score(y_train, y_pred_train, multioutput='raw_values')
print("R2_train:", R2_score_train)

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("\nMSE_test:", mse)
R2_score = r2_score(y_test, y_pred, multioutput='raw_values')
print("R2_test:", R2_score)

# Plotting the real data vs predicted data
plt.figure(figsize=(12, 8))
for i in range(y_test.shape[1]):
    plt.subplot(2, 3, i+1)
    plt.scatter(y_test[:, i], y_pred[:, i], c='r', label='Test Data')
    plt.scatter(y_train[:, i], y_pred_train[:, i], c='b', label='Train Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Real Value')
    plt.ylabel('Predicted Value')
    plt.title('Random Forest\nReal vs. Predicted '+ f'{data.columns[i+15]}')
    plt.legend(fontsize=7)
    plt.annotate(f"MSE_train: {mse_train[i]:.3f}", xy=(0.65, 0.2), 
                 xycoords='axes fraction')
    plt.annotate(f"R2_train: {R2_score_train[i]:.2f}", xy=(0.65, 0.15), 
                 xycoords='axes fraction')
    plt.annotate(f"MSE_test: {mse[i]:.3f}", xy=(0.65, 0.10), 
                 xycoords='axes fraction')
    plt.annotate(f"R2_test: {R2_score[i]:.2f}", xy=(0.65, 0.05), 
                 xycoords='axes fraction')

plt.tight_layout()

