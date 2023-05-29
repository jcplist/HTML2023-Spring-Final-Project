import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.ensemble import HistGradientBoostingRegressor


train_data = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

# full_features = ["Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"]


numeric_features = ["Acousticness", "Comments", "Duration_ms", "Energy", "Instrumentalness", "Likes", "Loudness", "Speechiness", "Stream", "Tempo", "Valence", "Views"]
numeric_transformer = Pipeline(steps=[
    ('scaler', RobustScaler(unit_variance=True)),
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features)
    ]
)

regr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('regressor', SVR(gamma = 1, shrinking=True, cache_size=32786)) # 1.8016201290887537
    # ('regressor', SVR(shrinking=True, cache_size=32786)) # 1.909511664994947
    # ('regressor', HistGradientBoostingRegressor(random_state=0xCC12, loss='absolute_error', l2_regularization=1000)) # 1.7187722414593387
    ('regressor', HistGradientBoostingRegressor(random_state=0, loss='absolute_error', l2_regularization=1000))
])

# regr.fit(X_train, y_train)
# y_pred = regr.predict(X_train)
# Ein = sum([abs(yp - yt) for yp, yt in zip(y_pred, y_train)]) / len(y_train)
# print(Ein)

logfile = open('result.log', 'w')

n_fold = 30
l2regs = [i for i in range(50, 151, 1)]
Evals = []

for l2reg in tqdm(l2regs):
    regr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=0, loss='absolute_error', l2_regularization=l2reg))
    ])
    cv = cross_validate(regr, X_train, y_train, scoring='neg_mean_absolute_error', cv=n_fold, n_jobs=-1)
    scores = cv['test_score']
    Eval = -sum(scores) / n_fold
    print('----------------------------------', file=logfile)
    print(f'l2reg={l2reg}\nEval={Eval}', file=logfile)
    print(np.array2string(scores), file=logfile)
    Evals.append(Eval)

logfile.close()

plt.plot(l2regs, Evals)
plt.title('HistGradientBoostingRegressor l2reg Eval')
plt.xlabel('l2_regularization')
plt.ylabel('Eval')

plt.savefig('HistGradientBoostingRegressor-l2reg-Eval.png', dpi=300)