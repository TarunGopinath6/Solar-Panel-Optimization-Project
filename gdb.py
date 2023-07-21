import pandas as pd
import xgboost as xgb
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv("./output_source.csv")
# data = pd.read_csv("./output_newdata.csv")

# Define the input features and the target variable
features = ["UNIX_TIME"]
#target = "ANGLE_FULL"
target = "ANGLE"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Define the parameters for the XGBoost model
params = {
    "learning_rate": 0.1,
    "max_depth": 20,
    "objective": "reg:squarederror",
    "n_estimators": 100
}

# Create the XGBoost model
model = xgb.XGBRegressor(**params)

# Train the model on the training data
model.fit(X_train, y_train)
# Evaluate the model on the testing data

#new_data = pd.DataFrame({'UNIX_TIME': [1591795800]})
#y_pred = model.predict(new_data)

y_pred = model.predict(X_test)
print(X_test.head())

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R-squared:", r2)

joblib.dump(model, 'model(1).joblib')

outs = pd.DataFrame()
outs['UNIX_TIME'] = pd.Series(X_test['UNIX_TIME'].values)
#outs['ANGLE_FULL'] = pd.Series(y_pred)
outs['ANGLE'] = pd.Series(y_pred)

outs.to_csv('model(1).csv', encoding='utf-8')