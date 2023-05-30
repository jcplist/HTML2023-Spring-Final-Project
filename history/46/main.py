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

# X_tran = preprocessor.fit_transform(X_train)
# print(preprocessor.get_feature_names_out())
# print(len(ordinal_features))
# print(X_tran.shape)
# print(X_tran)
# exit(0)

verbose = 0
lucky_cucumber = 0xCC12
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0, penalty='l2', alpha=0.001, max_iter=1000, tol=0.001, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
    # ('regressor', HistGradientBoostingRegressor(loss='absolute_error', l2_regularization=100000, random_state=lucky_cucumber, verbose=verbose))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_train)

# for yp, yt in zip(y_pred, y_train):
#     print(yp, yt)

Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_train)]) / len(y_train)
print(f'Ein = {Ein}')

folds = 8
cv = cross_validate(model, X_train, y_train, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)
scores = cv['test_score']
Eval = -sum(scores) / folds

# print(-scores)
print(f'Eval = {Eval}')

model.fit(X_train, y_train)
y_test = model.predict(X_test)
y_test = [y-0.5 if y < 4.5 else y+0.5 for y in y_test]
submit(y_test)