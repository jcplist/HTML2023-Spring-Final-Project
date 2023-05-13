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

for i in range(5):
    print("Eval:", Eval(i), f"({i})")

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

p_labels, q, qq = svm_predict([], test_x, regr, "-q")

submit(p_labels)