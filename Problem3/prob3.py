import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb
from sklearn.preprocessing import StandardScaler


plt.style.use(style="ggplot")

df = pd.read_csv('weather.csv', index_col=0)
df.head()

pearsoncorr = df.corr(method='pearson')
print(pearsoncorr)

sb.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

print(df['Humidity'].isnull().sum())


df.plot.scatter(x='Temperature (C)', y='Humidity')
plt.show()