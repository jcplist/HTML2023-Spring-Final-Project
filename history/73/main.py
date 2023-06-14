csvs = ["history/53/predict_1685715792.3907816.csv", "history/55/predict_1686326948.0414603.csv", "history/56/predict_1686403019.549613.csv", "history/57/predict_1686403669.154402.csv", "history/58/predict_1686405819.9889433.csv", "history/60/predict_1686408841.4744825.csv", "history/70/predict_1686701206.0068889.csv", "history/62/predict_1686410361.2050014.csv", "history/63/predict_1686482801.399256.csv"]
weights = [1, 1, 1, 1, 1, 1, 5, 1, 5]

import numpy as np
import pandas as pd
from submit import submit
a = np.zeros(6315)

for csv, w in zip(csvs, weights):
    b = pd.read_csv(csv).iloc[:,1].values
    b = (b - 4.5) * 1.2 + 4.5
    b[b > 7.5] += 0.2
    b[b < 1.5] -= 0.2
    a += b * w
a /= sum(weights)

submit(a)
