import numpy as np 
import pandas as pd
import math
from tqdm import tqdm
import pickle

from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from submit import submit

continuous_features = ["Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo"]
discrete_features = ["Duration_ms", "Views", "Likes", "Stream", "Comments"]
ordinal_features = continuous_features + discrete_features
ordinal_features = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Duration_ms", "Key"]
categorical_features = ["Composer", "Key", "Artist", "Album_type", "Licensed", "official_video"]
categorical_features = ["Composer"]

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('polynomial', PolynomialFeatures(degree=1, include_bias=False)),
    # ('scaler', RobustScaler(quantile_range = (25.0, 75.0), unit_variance=True))
    # ('scaler', StandardScaler())
    # ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', ordinal_transformer, ordinal_features),
    ('categorical', categorical_transformer, categorical_features)
])


lucky_cucumber = 0xCC12
df = pd.read_csv('train.csv')
df = shuffle(df, random_state=lucky_cucumber)

train_num = 500
X_raw_test = pd.read_csv('train.csv').iloc[:,1:]
X_train = np.array(preprocessor.fit_transform(X_raw_test.iloc[:train_num, :]))
X_val = np.array(preprocessor.fit_transform(X_raw_test.iloc[train_num:, :]))
y_raw = pd.read_csv('train.csv').iloc[:,0]
y_train = np.array([-1 if y < 4.5 else 1 for y in y_raw[:train_num]]).astype(int)
X_raw_test = pd.read_csv('test.csv')
X_test = np.array(preprocessor.fit_transform(X_raw_test))


# ADABoost

def sign(x): 
    return 1 if x >= 0 else -1

X_trainT = X_train.transpose()

d, n = np.shape(X_trainT)

X = np.ndarray((0, n))
y = np.ndarray((0, n))
indices = np.ndarray((0, n), dtype=int)
thetas = np.ndarray((0, n-1))

for i in range(d):
    sort_idx = np.argsort(X_trainT[i])
    Xi = X_trainT[i][sort_idx]
    indices = np.vstack((indices, sort_idx))
    X = np.vstack((X, Xi))
    y = np.vstack((y, y_train[sort_idx]))
    
    theta = [(Xi[idx+1]+Xi[idx])/2 for idx in range(n-1)]
    thetas = np.vstack((thetas, theta))

min_Ein = np.inf
max_Ein = -np.inf

T = 10000
S = [-1, 1]
u = np.full(n, 1/n)
gs, a = [], []
for t in tqdm(range(T)):
    best_Ein_u = np.inf
    best_g = (-1, 0, -np.inf)

    for s in S:
        for i in range(d):
            Ein = np.sum(y[i] < 0)
            Ein_u = 0
            for idx in range(n):
                Ein_u += u[indices[i][idx]] * (s * y[i][idx] < 0)
            
            if Ein_u < best_Ein_u:
                best_Ein_u = Ein_u
                best_g = (s, i, -np.inf)
            
            bound = 0
            for theta in thetas[i]:
                while bound < n and X[i][bound] < theta:
                    Ein += 1 if s * y[i][bound] > 0 else -1
                    Ein_u += u[indices[i][bound]] * (1 if s * y[i][bound] > 0 else -1)
                    bound += 1
                
                if Ein_u < best_Ein_u:
                    best_Ein_u = Ein_u
                    best_g = (s, i, theta)
    
    et = best_Ein_u / np.sum(u)
    dt = math.sqrt((1 - et) / et)

    s, i, theta = best_g
    Ein = 0
    for idx in range(n):
        if s * (X[i][idx] - theta) * y[i][idx] < 0:
            u[indices[i][idx]] *= dt
            Ein += 1
        else: 
            u[indices[i][idx]] /= dt 

    min_Ein = min(min_Ein, Ein)
    max_Ein = max(max_Ein, Ein)
    gs.append(best_g)
    a.append(math.log(dt))

    if (t+1) % (100) == 0:
        print(f"Iteration {t+1}\ns={s},\ti={i}\ttheta={theta}")
        print(f"Ein_u={best_Ein_u}\tEin={Ein}")
        print(f"min_Ein={min_Ein}, {min_Ein/n}\nmax_Ein={max_Ein}, {max_Ein/n}")
        print("-------------------------------", flush=True)

# END OF TRAIN

print(f"min_Ein_cnt={min_Ein},\t\tmin_Ein={min_Ein/n}")
print(f"max_Ein_cnt={max_Ein},\tmax_Ein={max_Ein/n}")

filepath = 'AdaBoost.model'
with open(filepath, 'wb') as file:
    pickle.dump(gs, file)
    pickle.dump(a, file)
print("model stored in ./" + filepath)

# filepath = 'AdaBoost.model'
# with open(filepath, 'rb') as file:
#     gs = pickle.load(file)
#     a = pickle.load(file)
# print("model read from ./" + filepath)

n = len(y_raw[:train_num])
predict = [sign(np.sum([at * sign(s * (x[i] - theta)) for at, (s, i, theta) in zip(a, gs)])) for x in X_train]
y_pred = [2.25 if y == -1 else 6.75 for y in predict]
Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_raw[:train_num])]) / n
print(f"Ein={Ein}")


n = len(y_raw[train_num:])
predict = [sign(np.sum([at * sign(s * (x[i] - theta)) for at, (s, i, theta) in zip(a, gs)])) for x in X_val]
y_pred = [2.25 if y == -1 else 6.75 for y in predict]
Eval = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_raw[train_num:])]) / n
print(f"Eval={Eval}")


predict = [sign(np.sum([at * sign(s * (x[i] - theta)) for at, (s, i, theta) in zip(a, gs)])) for x in X_test]
y_pred = [2.25 if y == -1 else 6.75 for y in predict]
# submit(y_pred)
