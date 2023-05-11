from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
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
used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer", "Artist"]

# Key: [0, 10]  Energy: [0, 1]  Tempo: [0, 243]  Valence: [0, 1]  Speechiness: [0, 0.96]

aa = set()
bb = set()

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
        x[i][8] = aa.index(x[i][8])

    for i in range(len(x)):
        x[i][9] = bb.index(x[i][9])
    return np.array(x)

train_y, train_x = get_train(used_entry)

for i in range(len(train_x)):
    aa.add(train_x[i][8])
aa = list(aa)
aa.pop(aa.index(np.nan))
aa.sort()
aa.append(np.nan)

for i in range(len(train_x)):
    bb.add(train_x[i][9])
bb = list(bb)
bb.pop(bb.index(np.nan))
bb.sort()
bb.append(np.nan)

#train_x = mean_imputation(train_x)
train_x = scaling(train_x)

rng = np.random.default_rng(seed=1987)
rng.shuffle(train_x)
rng = np.random.default_rng(seed=1987)
rng.shuffle(train_y)

regr = HistGradientBoostingRegressor(random_state=0, loss='absolute_error', categorical_features=[8, 9]).fit(train_x, train_y)

vy = regr.predict(train_x)

print("Ein:", sum([abs(train_y[i] - vy[i]) for i in range(len(vy))]) / len(vy))

def Eval (i):
    n = len(train_y)
    vxtr = np.concatenate((train_x[:(n // 5 * i)], train_x[(n // 5 * (i + 1)):]))
    vytr = np.concatenate((train_y[:(n // 5 * i)], train_y[(n // 5 * (i + 1)):]))
    vxt = train_x[(n // 5 * i):(n // 5 * (i + 1))]
    vyt = train_y[(n // 5 * i):(n // 5 * (i + 1))]

    vregr = HistGradientBoostingRegressor(random_state=0, loss='absolute_error', categorical_features=[8, 9]).fit(vxtr, vytr)
    vyp = vregr.predict(vxt)
    return sum([abs(vyt[i] - vyp[i]) for i in range(len(vyp))]) / len(vyp)

#with Pool(5) as p:
#    print(p.map(Eval, [0, 1, 2, 3, 4]))

for i in range(5):
    print("Eval:", Eval(i), f"({i})")

'''
Ein: 1.3946392422369074
Eval: 1.6188397119562272 (0)
Eval: 1.6157684869297257 (1)
Eval: 1.6382073075040398 (2)
Eval: 1.6095555652883873 (3)
Eval: 1.6492905756223122 (4)
'''

test_x = get_test(used_entry)
#test_x = mean_imputation(test_x)
test_x = scaling(test_x)

p_labels = regr.predict(test_x)

submit(p_labels)