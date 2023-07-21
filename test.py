import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import datetime

data = pd.read_csv("./energy_data.csv", header=0)
data = pd.DataFrame(data)

data['DATE'] = [datetime.datetime.strptime(x, "%d-%m-%Y %H:%M").date() 
                    for x in data['DATE_TIME']]
days = data['DATE'].unique()

angleData = pd.Series(dtype='float64')
angleFullData = pd.Series(dtype='float64')

data['UNIX_TIME'] = [int(datetime.datetime.timestamp(datetime.datetime.
                strptime(x, "%d-%m-%Y %H:%M"))) for x in data['DATE_TIME']]

for idx, day in enumerate(days):
    day_data = data[data['DATE'] == day]
    
    x = day_data.index
    y = day_data['DC_POWER']

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    
    angle = pd.Series(Gauss(x, 90, popt[1], popt[2]))
    
    maxAngleIndex = angle.idxmax()  
    topHalfAngle = 180 - angle[maxAngleIndex:]
    bottomHalfAngle = angle[:maxAngleIndex]
    angleFull = pd.concat([bottomHalfAngle, topHalfAngle])

    angleData = pd.concat([angleData, angle])
    angleFullData = pd.concat([angleFullData, angleFull])

    plt.subplot(6, 6, idx+1)
    plt.plot(x, y, '-b', label='data')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.plot(x, angle, '-g', label='angle')
    plt.plot(x, angleFull, '-y', label='Full angle')
    

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.show()
    
data['ANGLE'] = pd.Series(angleData.values)
data['ANGLE_FULL'] = pd.Series(angleFullData.values)
data.head()
data.to_csv('output.csv', encoding='utf-8')
