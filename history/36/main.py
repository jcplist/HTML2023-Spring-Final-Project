from parse_data import get_train, get_test
from imputation import mean_imputation
from submit import submit
from sklearn.ensemble import HistGradientBoostingClassifier
from multiprocessing import Pool
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)


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
train_y = np.array(train_y)

for i in range(len(train_x)):
    aa.add(train_x[i][Composer_index])
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

def neural (y, x):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1987)
    m = Sequential()
    m.add(Dense(1000, kernel_initializer='lecun_normal', activation='selu'))
    m.add(Dropout(0.2))
    m.add(Dense(500, kernel_initializer='lecun_normal', activation='selu'))
    m.add(Dropout(0.2))
    m.add(Dense(200, kernel_initializer='lecun_normal', activation='selu'))
    m.add(Dropout(0.2))
    m.add(Dense(10, kernel_initializer='lecun_normal', activation='selu'))
    m.add(Dropout(0.2))
    m.add(Dense(units = 1))
    m.compile(optimizer='rmsprop', loss='mae')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights = True)
    m.fit(X_train, y_train, validation_data = (X_test, y_test), callbacks=[es], epochs=5000, batch_size=50, verbose=1)
    return m

regr = neural(train_y, train_x)
vy = regr.predict(train_x)
print("Ein:", mean_absolute_error(vy, train_y)) # 1.66

def Eval (i):
    vxtr = np.concatenate((train_x[:(n // 5 * i)], train_x[(n // 5 * (i + 1)):]))
    vytr = np.concatenate((train_y[:(n // 5 * i)], train_y[(n // 5 * (i + 1)):]))
    vxt = train_x[(n // 5 * i):(n // 5 * (i + 1))]
    vyt = train_y[(n // 5 * i):(n // 5 * (i + 1))]
    vregr = neural(vytr, vxtr)
    vyp = vregr.predict(vxt)
    return mean_absolute_error(vyt, vyp)

#for i in range(5):
#    print("Eval:", Eval(i), f"({i})")

test_x = get_test(used_entry)
test_x = scaling(test_x)
test_x = mean_imputation(test_x)

ty = regr.predict(test_x)
p_labels = [a[0] for a in ty]

submit(p_labels)
