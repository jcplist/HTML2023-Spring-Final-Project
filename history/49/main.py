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
from sklearn.impute import KNNImputer
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

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    # ('imputer', KNNImputer(n_neighbors=5)),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
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
    # ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0, penalty='l2', alpha=0.001, max_iter=1000, tol=0.001, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
    ('regressor', HistGradientBoostingRegressor(loss='absolute_error', l2_regularization=100000, random_state=lucky_cucumber, verbose=verbose))
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
y_test1 = model.predict(X_test) # 42

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

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
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
y_test2 = model.predict(X_test)
y_test2 = [y-0.5 if y < 4.5 else y+0.5 for y in y_test2]

from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from libsvm.svmutil import *
from multiprocessing import Pool
import numpy as np
from math import sqrt

#used_entry = ["Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Views", "Likes", "Stream", "Comments"]

#1, Key, Speechiness^0.25, Energy, Valence 2.017 & <= 2.1
#1, instrumentalness^{1/16}, Speechiness^{1/4}, Energy, Valence 2.005 & <= 2.09
#1, instrumentalness^{1/16}, Speechiness^{1/4}, Energy^{1/2}, Valence 2 & <= 2.07
#1, instrumentalness^{1/16}, Speechiness^{1/4}, Energy^{1/2}, Valence, Acousticness^{1/16} 1.922 & <= 2.01
#1, instrumentalness^{1/16}, Speechiness^{1/4}, Energy^{1/2}, Valence, Acousticness^{1/16}, Liveness^{1/16} 1.905 & <= 2.001
#1, instrumentalness, Speechiness, Energy, Valence, Acousticness, Liveness (QuantileTransformer) 1.911 & <= 20.23
#1, instrumentalness^{1/16}, Speechiness^{1/4}, Energy^{1/2}, Valence, Acousticness^{1/16}, Liveness^{1/16}, Key/10 1.906 & <= 2.01
#1, instrumentalness^{1/16}, Speechiness^{1/4}, Energy^{1/2}, Valence, Acousticness^{1/16}, Liveness^{1/16}, Tempo/243 1.774 & <= 1.88
#used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer", "Artist"]
used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer"]
# Key: [0, 10]  Energy: [0, 1]  Tempo: [0, 243]  Valence: [0, 1]  Speechiness: [0, 0.96]

#experiment on l2_regularization = 1000
#importance Instrumentalness: 0.18
#importance Speechiness 0.18
#importance Energy 0.18
#importance Valence 0.12
#importance Acousticness 0.02
#importance Liveness 0.02
#importance Tempo 0.09
#importance Key -0.03 (But Eval is larger)
#importance Composer 0.06
#importance Artist 0.01 (Notice that Eval is smaller if we do not use Artist)

def mapping (y):
    for i in range(len(y)):
        y[i] = float(min(max((round(y[i])), 0), 9))
    return y

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
        composer_t = aa.index(x[i][8])
        x[i].pop()
        for _ in range(len(aa)):
            x[i].append(0)
        x[i][composer_t + 8] = 1
    return np.array(x)

train_y, train_x = get_train(used_entry)

for i in range(len(train_x)):
    aa.add(train_x[i][8])
aa = list(aa)
aa.pop(aa.index(np.nan))
aa.sort()
aa.append(np.nan)

train_x = scaling(train_x)
train_x = mean_imputation(train_x)

rng = np.random.default_rng(seed=1987)
rng.shuffle(train_x)
rng = np.random.default_rng(seed=1987)
rng.shuffle(train_y)

regr = svm_train(train_y, train_x, "-s 3 -t 2 -c 10 -g 0.5 -e 0.00001 -h 0 -m 8192 -q")

vy, q, qq = svm_predict(train_y, train_x, regr, "-q")
vy = mapping(vy)
print("Ein:", sum([abs(train_y[i] - vy[i]) for i in range(len(vy))]) / len(vy))

def Eval (i):
    n = len(train_y)
    vxtr = np.concatenate((train_x[:(n // 5 * i)], train_x[(n // 5 * (i + 1)):]))
    vytr = np.concatenate((train_y[:(n // 5 * i)], train_y[(n // 5 * (i + 1)):]))
    vxt = train_x[(n // 5 * i):(n // 5 * (i + 1))]
    vyt = train_y[(n // 5 * i):(n // 5 * (i + 1))]

    vregr = svm_train(vytr, vxtr, "-s 3 -t 2 -c 10 -g 0.5 -e 0.00001 -h 0 -m 8192 -q")
    vyp, q, qq = svm_predict(vyt, vxt, vregr, "-q")
    vyp = mapping(vyp)
    return sum([abs(vyt[i] - vyp[i]) for i in range(len(vyp))]) / len(vyp)

#for i in range(5):
#    print("Eval:", Eval(i), f"({i})")

'''
Ein: 1.605940594059406
Eval: 1.691322073383809 (0)
Eval: 1.7082119976703554 (1)
Eval: 1.7015142690739662 (2)
Eval: 1.6476412347117064 (3)
Eval: 1.672976121141526 (4)
'''

test_x = get_test(used_entry)
test_x = scaling(test_x)
test_x = mean_imputation(test_x)

y_test3, q, qq = svm_predict([], test_x, regr, "-q")

p_labels = [(y_test1[i] + y_test2[i] + y_test3[i]) / 3 for i in range(len(y_test1))]

"""
ADD -0.5 in submit 

Line 15: v = round(v) - 0.5
"""

submit(p_labels)