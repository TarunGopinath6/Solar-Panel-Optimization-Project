import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load('rf_model.joblib')
data = pd.DataFrame({
    'UNIX_TIME': 1680753728,
    'DC_POWER' : 23.89
}, index = [0])
predictions = model.predict(data)
print(predictions)