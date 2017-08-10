### Setup ###


import os
import hashlib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
from six.moves import urllib
from matplotlib import cm as cm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(42) # Makes outputs stable across runs

class DataFrameSelector(BaseEstimator, TransformerMixin): 
	"""Custom transformer for Pandas DataFrames (for Scikit Learn compatibility)"""
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names 
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values


### Fetching the Data ###


def load_wine_quality_data(wine_quality_path="winequality-red.csv"):
	"""Returns a Pandas DataFrame object containing formatted wine quality data"""
	csv_path = os.path.abspath(wine_quality_path) 
	return pd.read_csv(csv_path, sep=';')

wine = load_wine_quality_data()

print("Top five rows of DataFrame:")
print(wine.head())

print("")

print("Brief description of DataFrame:")
wine.info()

print("")

print("Number of instances per discrete quality score:")
print(wine["quality"].value_counts())

print("")

print("Summary of the features:")
print(wine.describe())

# Plots a 50-bin histogram for each feature
wine.hist(bins=50, figsize=(20,15))
plt.show()

# NOTE: The concentration on 5 and 6 in the wine quality distribution. This left-skew 
# will skew predictions towards 5 and 6, making other scores difficult to predict.

# NOTE: Some features have a tail-heavy distribution, so they may need to be transformed
# (e.g., by computing their logarithm).


### Build a Test Set ###


def test_set_check(identifier, test_ratio, hash):
	"""Checks if the last byte of an instance's hash is less than a test percentage of 256 bytes.
	   This ensures that the test set will be consistent across multiple runs."""
	return hash(np.int64(identifier)).digest()[-1] < (256 * test_ratio)

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5): 
	"""Assigns data to the test set based on a hash check."""
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
	return data.loc[~in_test_set], data.loc[in_test_set]

wine_with_id = wine.reset_index() # Adds an index column
train_set, test_set = split_train_test_by_id(wine_with_id, 0.2, "index") # 80% of data for training, 20% for testing

# NOTE: For the most important feature(s), apply stratified sampling.


### Understand the Data ###


wine = train_set.copy() # Excluding test_set avoids data snooping bias

"""
# Consolidate related features
wine["free SO2 / total SO2"] = wine["free sulfur dioxide"] / wine["total sulfur dioxide"]
wine["volatile acidity / fixed acidity"] = wine["volatile acidity"] / wine["fixed acidity"]
"""

# Check Spearman coefficients between each feature and quality
spear_corr_matrix = wine.corr(method='spearman') # Spearman's method b/c quality is an ordinal variable
print("Spearman correlations between all features and quality:")
print(spear_corr_matrix["quality"].sort_values(ascending=False))

print("")

# Plot a heatmap of Pearson coefficients b/w each feature
pear_corr_matrix = wine.corr(method='pearson') # TODO: Exclude quality
sns.heatmap(pear_corr_matrix, xticklabels=pear_corr_matrix.columns, yticklabels=pear_corr_matrix.columns)

"""
# Plots a scatter matrix between each feature
scatter_matrix(wine, alpha=0.1, figsize=(35, 35), diagonal='hist')
"""

# Plot the most positively correlated features
print("Scatterplot for alcohol and quality:")
wine.plot(kind="scatter", x="alcohol", y="quality", alpha=0.1)
plt.show()

print("")

# Plot the most negatively correlated features
print("Scatterplot for volatile acidity and quality:")
wine.plot(kind="scatter", x="volatile acidity", y="quality", alpha=0.1)
plt.show()


### Prepare the Data ###


wine = train_set.drop("quality", axis=1)
wine_labels = train_set["quality"].copy()

pipeline = Pipeline([
	('selector', DataFrameSelector(list(wine))),
	('std_scaler', StandardScaler())
])

wine_prepared = pipeline.fit_transform(wine)


### Selecting and Training a Model ###


def display_scores(scores):
	"""Displays cross-validation scores, the mean, and standard deviation"""
	print("Scores:", scores)
	print("Mean:", scores.mean())
	print("Standard Deviation:", scores.std())

lin_reg = LinearRegression()
lin_reg.fit(wine_prepared, wine_labels)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(wine_prepared, wine_labels)

forest_reg = RandomForestRegressor()
forest_reg.fit(wine_prepared, wine_labels)

svr_rbf = SVR(kernel='rbf') # Prone to overfitting
svr_rbf.fit(wine_prepared, wine_labels)

svr_lin = SVR(kernel='linear')
svr_lin.fit(wine_prepared, wine_labels)

"""
some_data_prepared = pipeline.transform(wine.iloc[:5])
some_labels = wine_labels.iloc[:5]
print("Predictions:\t", np.round(lin_reg.predict(some_data_prepared), decimals=0))
print("Labels:\t\t", list(some_labels))
"""

# K-Fold Cross-Validation (with 5 evaluation scores)
lin_scores = cross_val_score(lin_reg, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Linear Regression Performance:")
display_scores(lin_rmse_scores) # Mean: 0.670370703554 STD: 0.0203083877576

print("")

tree_scores = cross_val_score(tree_reg, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
tree_rmse_scores = np.sqrt(-tree_scores)
print("Decision Tree Performance:")
display_scores(tree_rmse_scores) # Mean: 0.929498932565 STD: 0.061981747745

print("")

forest_scores = cross_val_score(forest_reg, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-forest_scores)
print("Random Forest Performance:")
display_scores(forest_rmse_scores) # Mean: 0.685924713092 STD: 0.0258007542247

print("")

svr_rbf_scores = cross_val_score(svr_rbf, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
svr_rbf_rmse_scores = np.sqrt(-svr_rbf_scores)
print("Support Vector Regression (Radial) Performance:")
display_scores(svr_rbf_rmse_scores) # Mean: 0.655513263441 STD: 0.0310762012316

print("")

svr_lin_scores = cross_val_score(svr_lin, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
svr_lin_rmse_scores = np.sqrt(-svr_lin_scores)
print("Support Vector Regression (Linear) Performance:")
display_scores(svr_lin_rmse_scores) # Mean: 0.670212855876 STD: 0.0158305891655

print("")


### Fine-Tuning ###


svr_rbf_param_grid = [{'C': [0.1, 0.5, 1, 2, 3, 5, 10], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], 'epsilon': [0.01, 0.05, 0.1, 0.5, 1]}]
svr_lin_param_grid = [{'C': [0.1, 0.5, 1, 2, 3, 5, 10]}]

svr_rbf_grid_search = GridSearchCV(svr_rbf, svr_rbf_param_grid, cv=5, scoring='neg_mean_squared_error')
svr_lin_grid_search = GridSearchCV(svr_lin, svr_lin_param_grid, cv=5, scoring='neg_mean_squared_error')

svr_rbf_grid_search.fit(wine_prepared, wine_labels)
svr_lin_grid_search.fit(wine_prepared, wine_labels)

print("Best parameters for SVR (Radial): ", svr_rbf_grid_search.best_params_) # {'epsilon': 0.1, 'C': 1, 'gamma': 0.01}
print("Best parameters for SVR (Linear): ", svr_lin_grid_search.best_params_) # {'C': 0.1}

print("")

updated_svr_rbf_scores = cross_val_score(svr_rbf_grid_search.best_estimator_, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
updated_svr_rbf_rmse_scores = np.sqrt(-updated_svr_rbf_scores)
print("Support Vector Regression (Radial) Performance w/ Updated Parameters:")
display_scores(updated_svr_rbf_rmse_scores) # Mean: 0.645562191616 STD: 0.0156204929688

print("")

updated_svr_lin_scores = cross_val_score(svr_lin_grid_search.best_estimator_, wine_prepared, wine_labels, scoring="neg_mean_squared_error", cv=5)
updated_svr_lin_rmse_scores = np.sqrt(-updated_svr_lin_scores)
print("Support Vector Regression (Linear) Performance w/ Updated Parameters:")
display_scores(updated_svr_lin_rmse_scores) # Mean: 0.669521175482 STD: 0.0148426915748

print("")

final_model = svr_rbf_grid_search.best_estimator_

joblib.dump(final_model, "model.pkl")