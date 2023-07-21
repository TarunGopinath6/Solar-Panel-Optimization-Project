import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
from micromlgen import port

# Load dataset
df = pd.read_csv("output_source.csv")

# Split into features and target
X = df[["UNIX_TIME", "DC_POWER"]]
y = df["ANGLE"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on test set
y_pred = xgb_model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(xgb_model, 'xgb_model.joblib')

outs = pd.DataFrame()
outs['UNIX_TIME'] = pd.Series(X_test['UNIX_TIME'].values)
outs['ANGLE'] = pd.Series(y_pred)
outs.sort_values(by="UNIX_TIME", ascending=True)
outs.to_csv('xgb_model.csv', encoding='utf-8', index=False)

print("MSE:", mse)
print("R-squared:", r2)
print(port(xgb_model))