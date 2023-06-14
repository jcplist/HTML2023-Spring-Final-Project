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
from sklearn.feature_selection import RFE
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
regr = SGDRegressor(loss='epsilon_insensitive', epsilon=0.08, penalty='l2', alpha=1.2e-6, max_iter=1000, tol=0.06, shuffle=True, random_state=lucky_cucumber, verbose=verbose)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('selector', RFE(regr, n_features_to_select=42, step=1)),
    ('regressor', regr)
])

model.fit(X_train, y_train)
y_pred = model.predict(X_train)
Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_train)]) / len(y_train)
print(f'Ein = {Ein}')

y_pred = model.predict(X_val)
Eval = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_val)]) / len(y_val)
print(f'Eval = {Eval}')

y_test1 = model.predict(X_test)

from parse_data import get_train, get_test, get_val
from imputation import mean_imputation
from submit import submit
from libsvm.svmutil import *
import numpy as np
from math import sqrt

used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer"]

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

regr = svm_train(train_y, train_x, "-s 3 -t 2 -c 0.01 -g 0.5 -e 0.00001 -h 0 -m 8192 -q")

vy, q, qq = svm_predict(train_y, train_x, regr, "-q")
vy = mapping(vy)
print("Ein :", sum([abs(train_y[i] - vy[i]) for i in range(len(vy))]) / len(vy))

val_y, val_x = get_val(used_entry)
val_x = scaling(val_x)
val_x = mean_imputation(val_x)
vy, q, qq = svm_predict(val_y, val_x, regr, "-q")
vy = mapping(vy)
print("Eval:", sum([abs(val_y[i] - vy[i]) for i in range(len(vy))]) / len(vy))


test_x = get_test(used_entry)
test_x = scaling(test_x)
test_x = mean_imputation(test_x)

y_test2, q, qq = svm_predict([], test_x, regr, "-q")

y_test1 = np.array(y_test1)
y_test2 = np.array(y_test2)

submit((y_test1 + y_test2) / 2)
