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
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from submit import submit

pd_train = pd.read_csv('train.csv')
X_train = pd_train.iloc[:,1:]
y_train = pd_train.iloc[:,0]
X_val = pd.read_csv('val.csv').iloc[:,1:]
y_val = pd.read_csv('val.csv').iloc[:,0]
X_test = pd.read_csv('test.csv')

continuous_features = ["Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo"]
discrete_features = ["Duration_ms", "Views", "Likes", "Stream", "Comments"]
ordinal_features = continuous_features + discrete_features
# ordinal_features = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key"]
categorical_features = ["Composer", "Key", "Artist", "Album_type", "Licensed", "official_video"]
# categorical_features = ["Composer"]

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

y_train = y_train.values
y_val = y_val.values

verbose = 1
lucky_cucumber = 0xCC12
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0, penalty='l2', alpha=0.001, max_iter=1000, tol=0.001, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
    ('regressor', ElasticNetCV(random_state=lucky_cucumber, verbose=verbose, n_jobs=-1))
])

model.fit(X_train, y_train)
vy = model.predict(X_train)
print("Ein :", mean_absolute_error(y_train, vy))
vy = model.predict(X_val)
print("Eval:", mean_absolute_error(y_val, vy))
y_test = model.predict(X_test)

submit(y_test)
