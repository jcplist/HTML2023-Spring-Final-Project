from parse_data import get_train, get_test, get_val
from imputation import mean_imputation
from submit import submit
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.model_selection import cross_validate
import numpy as np
from math import sqrt

used_entry = ["Instrumentalness", "Speechiness", "Energy", "Valence", "Acousticness", "Liveness", "Tempo", "Key", "Composer", "Artist"]

Composer_index = 8
Artist_index = 9

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
        x[i][Composer_index] = aa.index(x[i][Composer_index])
        x[i][Artist_index] = bb.index(x[i][Artist_index])

    return x

train_y, train_x = get_train(used_entry)

for i in range(len(train_x)):
    aa.add(train_x[i][Composer_index])
aa = list(aa)
aa.pop(aa.index(np.nan))
aa.sort()
aa.append(np.nan)
for i in range(len(train_x)):
    bb.add(train_x[i][Artist_index])
bb = list(bb)
bb.pop(bb.index(np.nan))
bb.sort()
bb.append(np.nan)


train_x = scaling(train_x)

val_y, val_x = get_val(used_entry)
val_x = scaling(val_x)

regr = CatBoostRegressor(task_type="GPU", devices='0', random_state=0xCC12, loss_function='MAE', eval_metric = 'MAE', cat_features=[Composer_index, Artist_index], verbose = 0)#.fit(train_x, train_y, eval_set = (val_x, val_y), use_best_model = True)

summary = regr.select_features(
    train_x, train_y,
    eval_set=(val_x, val_y),
    features_for_select='0-9',
    num_features_to_select=8,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    plot=False
)

print(summary)

vy = regr.predict(train_x)

print("Ein:", mean_absolute_error(vy, train_y))

vy = regr.predict(val_x)
print("Eval:", mean_absolute_error(vy, val_y))

test_x = get_test(used_entry)
test_x = scaling(test_x)

y_test = regr.predict(test_x)

submit(y_test)
