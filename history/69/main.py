import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor

from submit import submit

X_train = pd.read_csv('train.csv').iloc[:,1:]
y_train = pd.read_csv('train.csv').iloc[:,0]
X_val = pd.read_csv('val.csv').iloc[:,1:]
y_val = pd.read_csv('val.csv').iloc[:,0]
X_test = pd.read_csv('test.csv')

continuous_features = ["Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo"]
discrete_features = ["Duration_ms", "Views", "Likes", "Stream", "Comments", "Key"]
ordinal_features = continuous_features + discrete_features
categorical_features = ["Composer", "Artist", "Album_type", "Licensed", "official_video"]

categorical_transformer = Pipeline(steps=[
    # ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

imputer = SimpleImputer(strategy='median')
scaler = MinMaxScaler()

for feat in ordinal_features:
    # training data
    col = X_train[feat].values.reshape(-1, 1)
    col = imputer.fit_transform(col)
    col = scaler.fit_transform(col)
    X_train[feat] = col
    # validating data
    col = X_val[feat].values.reshape(-1, 1)
    col = imputer.fit_transform(col)
    col = scaler.fit_transform(col)
    X_val[feat] = col
    # testing data
    col = X_test[feat].values.reshape(-1, 1)
    col = imputer.fit_transform(col)
    col = scaler.fit_transform(col)
    X_test[feat] = col

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', 'passthrough', ordinal_features),
    ('categorical', categorical_transformer, categorical_features)
])

verbose = 0
lucky_cucumber = 0xCC12
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0.12, penalty='l2', alpha=1.7e-5, max_iter=1000, tol=0.06, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_train)]) / len(y_train)
print(f'Ein = {Ein}')

y_pred = model.predict(X_val)
Eval = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_val)]) / len(y_val)
print(f'Eval = {Eval}')

y_test = model.predict(X_test)
submit(y_test)