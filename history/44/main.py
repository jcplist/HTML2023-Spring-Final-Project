import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate

from libsvm.svmutil import *

from imputation import mean_imputation
from parse_data import get_train, get_test
from submit import submit

"""
model 39
"""


X_train = pd.read_csv('train.csv').iloc[:,1:]
y_train = pd.read_csv('train.csv').iloc[:,0]
X_test = pd.read_csv('test.csv')

continuous_features = ["Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo"]
discrete_features = ["Duration_ms", "Views", "Likes", "Stream", "Comments"]
ordinal_features = continuous_features + discrete_features
ordinal_features = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo"]
categorical_features = ["Composer", "Key", "Artist", "Album_type", "Licensed", "official_video"]
categorical_features = ["Composer", "Key"]

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # ('imputer', KNNImputer(n_neighbors=5)),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=True)),
    # ('scaler', RobustScaler(quantile_range = (25.0, 75.0), unit_variance=True))
    ('scaler', StandardScaler())
    # ('scaler', MinMaxScaler())
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
# lucky_cucumber = 0x09902017
lucky_cucumber = 0xCC12
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0, penalty='l2', alpha=0.001, max_iter=1000, tol=0.0001, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
    # ('regressor', HistGradientBoostingRegressor(loss='absolute_error', l2_regularization=100000, random_state=lucky_cucumber, verbose=verbose))
])

model.fit(X_train, y_train)
# y_pred = model.predict(X_train)

# for yp, yt in zip(y_pred, y_train):
#     print(yp, yt)

# Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_train)]) / len(y_train)
# print(f'Ein = {Ein}')

# folds = 8
# cv = cross_validate(model, X_train, y_train, cv=folds, scoring='neg_mean_absolute_error', n_jobs=-1)
# scores = cv['test_score']
# Eval = -sum(scores) / folds

# print(-scores)
# print(f'Eval = {Eval}')

model.fit(X_train, y_train)
p_labels0 = model.predict(X_test)

"""
model 22
"""


used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer"]
used_entry2 = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer", "Loudness", "Duration_ms"]
Composer_index = 8

aa = set()

def scaling (x):

    for i in range(len(x)):
        x[i][0] = sqrt(sqrt(sqrt(sqrt(x[i][0]))))
        x[i][1] = sqrt(sqrt(x[i][1]))
        x[i][2] = sqrt(x[i][2])
        x[i][4] = sqrt(sqrt(sqrt(sqrt(x[i][4]))))
        x[i][5] = sqrt(sqrt(sqrt(sqrt(x[i][5]))))
        x[i][6] /= 243
        x[i][7] /= 10
    
    for i in range(len(x)):
        x[i][Composer_index] = aa.index(x[i][Composer_index])

    return np.array(x)

train_y, train_x = get_train(used_entry)

for i in range(len(train_x)):
    aa.add(train_x[i][Composer_index])
aa = list(aa)
aa.pop(aa.index(np.nan))
aa.sort()
aa.append(np.nan)

train_x = scaling(train_x)

rng = np.random.default_rng(seed=1987)
rng.shuffle(train_x)
rng = np.random.default_rng(seed=1987)
rng.shuffle(train_y)

regr = HistGradientBoostingRegressor(random_state=0, loss='absolute_error', categorical_features=[8], l2_regularization=1000).fit(train_x, train_y)

vy = regr.predict(train_x)

test_x = get_test(used_entry)
test_x = scaling(test_x)

test_x = test_x[2000:]

p_labels1 = regr.predict(test_x)

train_y, train_x = get_train(used_entry2)

train_x = scaling(train_x)

rng = np.random.default_rng(seed=1987)
rng.shuffle(train_x)
rng = np.random.default_rng(seed=1987)
rng.shuffle(train_y)

regr = HistGradientBoostingRegressor(random_state=0, loss='absolute_error', categorical_features=[8], l2_regularization=1000).fit(train_x, train_y)

test_x = get_test(used_entry2)
test_x = scaling(test_x)

test_x = test_x[:2000]

p_labels2 = regr.predict(test_x)

p_labels1 = np.concatenate((p_labels2, p_labels1))

def scaling2 (x):
    for i in range(len(x)):
        x[i][0] = sqrt(sqrt(sqrt(sqrt(x[i][0]))))
        x[i][1] = sqrt(sqrt(x[i][1]))
        x[i][2] = sqrt(x[i][2])
        x[i][4] = sqrt(sqrt(sqrt(sqrt(x[i][4]))))
        x[i][5] = sqrt(sqrt(sqrt(sqrt(x[i][5]))))
        x[i][6] /= 243
        x[i][7] /= 10
        composer_t = aa.index(x[i][8])
        x[i].pop()
        for _ in range(len(aa)):
            x[i].append(0)
        x[i][composer_t + 8] = 1
    return np.array(x)

train_y, train_x = get_train(used_entry)
train_x = scaling2(train_x)
train_x = mean_imputation(train_x)

rng = np.random.default_rng(seed=1987)
rng.shuffle(train_x)
rng = np.random.default_rng(seed=1987)
rng.shuffle(train_y)

regr = svm_train(train_y, train_x, "-s 3 -t 2 -c 10 -g 0.5 -e 0.00001 -h 0 -m 8192 -q")

test_x = get_test(used_entry)
test_x = scaling2(test_x)
test_x = mean_imputation(test_x)

p_labels2, q, qq = svm_predict([], test_x, regr, "-q")

print(sum([abs(p_labels2[i] - p_labels1[i]) for i in range(len(p_labels1))]) / len(p_labels1))

p_labels = p_labels1[:]

for i in range(len(p_labels1)):
    p_labels[i] = (p_labels0[i] + p_labels1[i] + p_labels2[i]) / 3

submit(p_labels)