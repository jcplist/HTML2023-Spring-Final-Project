import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from libsvm.svmutil import *

from submit import submit

X_train = pd.read_csv('train.csv').iloc[:,1:]
y_train = pd.read_csv('train.csv').iloc[:,0]
X_test = pd.read_csv('test.csv')

continuous_features = ["Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo"]
discrete_features = ["Duration_ms", "Views", "Likes", "Stream", "Comments"]
ordinal_features = continuous_features + discrete_features
# ordinal_features = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key"]
categorical_features = ["Composer", "Key", "Artist", "Album_type", "Licensed", "official_video"]
# categorical_features = ["Composer"]

def scale (x, feature):
    x.loc[:][feature].update(x.loc[:][feature].pow(0.5))
    return x

for _ in ["Instrumentalness", "Speechiness", "Energy", "Acousticness", "Liveness"]:
    X_train = scale(X_train, _)
    X_test = scale(X_test, _)
for _ in ["Instrumentalness", "Speechiness", "Acousticness", "Liveness"]:
    X_train = scale(X_train, _)
    X_test = scale(X_test, _)
for _ in ["Instrumentalness", "Acousticness", "Liveness"]:
    X_train = scale(X_train, _)
    X_test = scale(X_test, _)
for _ in ["Instrumentalness", "Acousticness", "Liveness"]:
    X_train = scale(X_train, _)
    X_test = scale(X_test, _)

for i in range(len(X_train)):
    if X_train.loc[i]["Artist"] == "Father John Misty":
        X_train.at[i, "Artist"] = np.nan

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    #('rscaler', RobustScaler(quantile_range = (25.0, 75.0), unit_variance=True)),
    ('scaler', StandardScaler())#,
    #('mmscaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', ordinal_transformer, ordinal_features),
    ('categorical', categorical_transformer, categorical_features)
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

y_train = y_train.values

regr = svm_train(y_train, X_train, "-s 3 -t 2 -c 0.1 -t 0 -e 0.00001 -h 0 -m 30000 -q")

y_pred, q, qq = svm_predict([], X_train, regr, "-q")
print(f'Ein = {mean_absolute_error(y_pred, y_train)}')


y_test, q, qq = svm_predict([], X_test, regr, "-q")
submit(y_test)
