import numpy as np
from sklearn.externals import joblib
from training import test_set, pipeline
from sklearn.metrics import mean_squared_error

svr_rbf = joblib.load("model.pkl")

X_test = test_set.drop("quality", axis=1)
y_test = test_set["quality"].copy()

X_test_prepared = pipeline.transform(X_test)

predictions = np.round(svr_rbf.predict(X_test_prepared), decimals=0)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Predictions: ", predictions)
print("RMSE: ", rmse)