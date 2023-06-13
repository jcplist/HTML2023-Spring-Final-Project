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
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     # ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0, penalty='l2', alpha=0.001, max_iter=1000, tol=0.001, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
#     ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=0, penalty='l2', alpha=0.0001, max_iter=1000, tol=0.1, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
#     # ('regressor', HistGradientBoostingRegressor(loss='absolute_error', l2_regularization=100000, random_state=lucky_cucumber, verbose=verbose))
#     # ('regressor', LinearSVR(loss='epsilon_insensitive', epsilon=0, C=0.010, tol=0.001, random_state=lucky_cucumber, verbose=verbose))
#     # ('regressor', SVR(kernel='rbf', gamma='scale', epsilon=0, C=0.1, tol=0.01, verbose=verbose))
# ])



epsilons = [0+i/100 for i in range(0, 21, 1)]
alphas = [1e-5+i/1e6 for i in range(-10, 10, 1)]
tols = [0.1+i/100 for i in range(-10, 10, 1)]
combs = itertools.product(epsilons, alphas, tols)

logfile = open('SGDregressor-parameter.log', 'w')
progress_bar = tqdm(total=len(epsilons)*len(alphas)*len(tols), desc='Progress')

best_Eval = np.inf
for (epsilon, alpha, tol) in combs:
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', SGDRegressor(loss='epsilon_insensitive', epsilon=epsilon, penalty='l2', alpha=alpha, max_iter=1000, tol=tol, shuffle=True, random_state=lucky_cucumber, verbose=verbose))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_train)]) / len(y_train)
    # print(f'Ein = {Ein}')

    y_pred = model.predict(X_val)
    Eval = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_val)]) / len(y_val)
    # print(f'Eval = {Eval}')

    print(f'(eps={epsilon}, alpha={alpha}, tol={tol}): Ein={Ein}, Eval={Eval}', file=logfile)
    if Eval < best_Eval:
        best_comb = (epsilon, alpha, tol)
        best_Eval = Eval
    
    progress_bar.update(1)


print('\n----------------------------------\n\nBest Parameters\n', file=logfile)
print(f'(eps={best_comb[0]}, alpha={best_comb[1]}, tol={best_comb[2]}): Eval={best_Eval}', file=logfile)
print(f'(eps={best_comb[0]}, alpha={best_comb[1]}, tol={best_comb[2]}): Eval={best_Eval}')

# y_test = model.predict(X_test)