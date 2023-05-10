from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from multiprocessing import Pool
import numpy as np

#used_entry = ["Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Views", "Likes", "Stream", "Comments"]

used_entry = ["Key", "Energy", "Tempo", "Valence", "Speechiness", "Acousticness"]

# Key: [0, 10]  Energy: [0, 1]  Tempo: [0, 243]  Valence: [0, 1]  Speechiness: [0, 0.96]

def scaling (x):
    for i in range(len(x)):
        x[i][0] /= 10
        x[i][2] /= 243
        x[i][4] /= 0.96
    return x

train_y, train_x = get_train(used_entry)
train_x = mean_imputation(train_x)
train_x = scaling(train_x)

regr = BaggingRegressor(estimator=SVR(kernel='rbf', gamma=10, C=1.2, tol=0.00001, cache_size=4096, shrinking=False), n_estimators=10, random_state=0).fit(train_x, train_y)

vy = regr.predict(train_x)

print("Ein:", sum([abs(train_y[i] - vy[i]) for i in range(len(vy))]) / len(vy)) # 1.7364887634821358

def Eval (i):
    n = len(train_y)
    vxtr = np.concatenate((train_x[:(n // 5 * i)], train_x[(n // 5 * (i + 1)):]))
    vytr = np.concatenate((train_y[:(n // 5 * i)], train_y[(n // 5 * (i + 1)):]))
    vxt = train_x[(n // 5 * i):(n // 5 * (i + 1))]
    vyt = train_y[(n // 5 * i):(n // 5 * (i + 1))]

    vregr = BaggingRegressor(estimator=SVR(kernel='rbf', gamma=10, C=1.2, tol=0.00001, cache_size=1024, shrinking=False), n_estimators=10, random_state=0).fit(vxtr, vytr)
    vyp = vregr.predict(vxt)
    return sum([abs(vyt[i] - vyp[i]) for i in range(len(vyp))]) / len(vyp)

with Pool(5) as p:
    print(p.map(Eval, [0, 1, 2, 3, 4]))

# [1.8004993065469754, 1.8798016565923736, 1.7805201487311468, 1.8763019480852896, 1.9394606771149492]

test_x = get_test(used_entry)
test_x = mean_imputation(test_x)
test_x = scaling(test_x)

p_labels = regr.predict(test_x)

submit(p_labels)