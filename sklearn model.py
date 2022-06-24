import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("USA_Housing.csv")

a = pd.plotting.scatter_matrix(df, hist_kwds={"bins": 10, "rwidth": 0.7})

my_df_cols = ['Avg. Area Income', 'Avg. Area House Age',
              'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
              'Area Population', 'Price', 'Address']
x = df[['Avg. Area Income', 'Avg. Area House Age',
        'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
        'Area Population']]
y = df["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

Ai = LinearRegression()
Ai.fit(x_train, y_train)
predictions = Ai.predict(x_test)
error = mean_absolute_error(y_test, predictions)
print(error)
print(np.min(predictions))
# plt.scatter(predictions, y_test)
# plt.xlabel("predictions")
plt.show()
