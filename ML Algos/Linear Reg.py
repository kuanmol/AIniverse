from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450]])
Y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)
print(y_pred)
