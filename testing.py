### Setup ###


import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from training import test_set, pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'font.size': 18})

svr_rbf = joblib.load("model.pkl")

X_test = test_set.drop(["quality", "citric acid", "pH"], axis=1)
y_test = test_set["quality"].copy()

X_test_prepared = pipeline.transform(X_test)

predictions = np.round(svr_rbf.predict(X_test_prepared), decimals=0)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("Predictions: ", predictions)
print("RMSE: ", rmse)

print("")

def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
	"""This function prints and plots the confusion matrix. Normalization
	   can be applied by setting `normalize=True`.
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
	plt.yticks(tick_marks, classes, fontsize=18)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized Confusion Matrix")
	else:
		print("Confusion Matrix, without normalization")

	print(cm)

	thresh = cm.max() / 2.0
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
			horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
	
	plt.tight_layout()
	plt.ylabel('True label', fontsize=18)
	plt.xlabel('Predicted label', fontsize=18)

cnf_matrix = confusion_matrix(y_test, predictions) # Compute confusion matrix

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=range(min(y_test), (max(y_test) + 1), 1))
plt.show()