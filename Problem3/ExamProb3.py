import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk import OrderedDict
from scipy import stats
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

feature_names = ['Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
                 'Visibility (km)', 'Loud Cover', 'Pressure (millibars)']
target_name = ['Temperature (C)']

X = df[feature_names]
Y = df[target_name]

X = X.apply(pd.to_numeric, errors='coerce')
Y = Y.apply(pd.to_numeric, errors='coerce')
X.fillna(0, inplace=True)
Y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
error_metric = mean_squared_error(y_pred=y_pred_test, y_true=y_test)
print("mean squared of this model is: ", error_metric)

new_data = OrderedDict([('Apparent Temperature (C)', 7.3),
                        ('Humidity', .98),
                        ('Wind Speed (km/h)', 13),
                        ('Wind Bearing (degrees)', 240),
                        ('Visibility (km)', 15),
                        ('Loud Cover', 0),
                        ('Pressure (millibars)', 1015),])


new_data = pd.Series(new_data).values.reshape(1, -1)
print(model.predict(new_data))
