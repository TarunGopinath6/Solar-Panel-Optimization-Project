import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("xgb_model.csv")
index = [i for i in range(len(data))]
plt.plot(index, data['ANGLE'])
plt.show()