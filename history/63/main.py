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

p_labels, q, qq = svm_predict([], test_x, regr, "-q")

submit(p_labels)
