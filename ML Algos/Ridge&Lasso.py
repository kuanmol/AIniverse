from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

X = np.array([[1400], [1600], [1700], [1875], [1100], [1550], [2350], [2450]])
Y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
ridge_predictions=ridge_model.predict(X_test)
ridge_mse=mean_squared_error(y_test,ridge_predictions)
print("Ridge MSE: ",ridge_mse)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions=lasso_model.predict(X_test)
lasso_mse=mean_squared_error(y_test,lasso_predictions)
print("Lasso MSE: ",lasso_mse)
