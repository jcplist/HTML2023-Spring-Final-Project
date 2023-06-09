import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

from submit import submit

X_train = pd.read_csv('train.csv').iloc[:,1:]
y_train = pd.read_csv('train.csv').iloc[:,0]
X_val = pd.read_csv('val.csv').iloc[:,1:]
y_val = pd.read_csv('val.csv').iloc[:,0]
X_test = pd.read_csv('test.csv')

continuous_features = ["Energy","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo"]
discrete_features = ["Duration_ms", "Views", "Likes", "Stream", "Comments"]
ordinal_features = continuous_features + discrete_features
# ordinal_features = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key"]
categorical_features = ["Composer", "Key", "Artist", "Album_type", "Licensed", "official_video"]
# categorical_features = ["Composer"]

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    #('rscaler', RobustScaler(quantile_range = (25.0, 75.0), unit_variance=True)),
    ('scaler', StandardScaler())#,
    #('mmscaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('ordinal', ordinal_transformer, ordinal_features),
    ('categorical', categorical_transformer, categorical_features)
])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

y_train = y_train.values
y_val = y_val.values

m = Sequential()
m.add(Dense(1000, input_shape=(X_train.shape[1],), activation='linear'))
m.add(Dropout(0.5))
m.add(Dense(500, activation='linear'))
m.add(Dropout(0.5))
m.add(Dense(250, activation='linear'))
m.add(Dropout(0.5))
m.add(Dense(1, activation='linear'))
m.compile(optimizer='rmsprop', loss='mae')
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights = True)
m.fit(X_train, y_train, validation_data = (X_val, y_val), callbacks=[es], epochs=5000, batch_size=50, verbose=1)

ty = m.predict(X_test)
y_test = [a[0] for a in ty]
submit(y_test)

