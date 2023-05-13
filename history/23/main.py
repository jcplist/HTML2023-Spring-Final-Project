from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from libsvm.svmutil import *
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
#used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer", "Artist"]
used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer"]
used_entry2 = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer", "Loudness", "Duration_ms"]
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

vy2, q, qq = svm_predict(train_y, train_x, regr, "-q")

test_x = get_test(used_entry)
test_x = scaling2(test_x)
test_x = mean_imputation(test_x)

p_labels2, q, qq = svm_predict([], test_x, regr, "-q")

print("diff:", sum([abs(p_labels2[i] - p_labels1[i]) for i in range(len(p_labels1))]) / len(p_labels1))

phi_x = []
for i in range(len(vy)):
    phi_x.append([vy[i], vy2[i]])

regr = svm_train(train_y, phi_x, "-s 3 -t 0 -e 0.00001 -m 8192")

phi_x = []
for i in range(len(p_labels1)):
    phi_x.append([p_labels1[i], p_labels2[i]])

p_labels, q, qq = svm_predict([], phi_x, regr, "")

submit(p_labels)