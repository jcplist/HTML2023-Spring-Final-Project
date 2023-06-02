import pandas as pd
import numpy as np
from io import StringIO

def get_train (used_entry):
    with open("train.csv", "r") as f:
        train_data = f.read()
    train_data = pd.read_csv(StringIO(train_data))

    train_x = [[] for _ in range(len(train_data))]
    for entry in used_entry:
        for i in range(len(train_data)):
            train_x[i].append(train_data[entry][i])
    train_y = [_ for _ in train_data['Danceability']]

    return (train_y, train_x)

def get_val (used_entry):
    with open("val.csv", "r") as f:
        train_data = f.read()
    train_data = pd.read_csv(StringIO(train_data))

    train_x = [[] for _ in range(len(train_data))]
    for entry in used_entry:
        for i in range(len(train_data)):
            train_x[i].append(train_data[entry][i])
    train_y = [_ for _ in train_data['Danceability']]

    return (train_y, train_x)

def get_test (used_entry):
    with open("test.csv", "r") as f:
        test_data = f.read()
    test_data = pd.read_csv(StringIO(test_data))

    test_x = [[] for _ in range(len(test_data))]
    for entry in used_entry:
        for i in range(len(test_data)):
            test_x[i].append(test_data[entry][i])

    return test_x

def csv_to_dict(filename):
    result_dict = {}
    df = pd.read_csv(filename)
    for column in df.columns:
        try:
            result_dict[column] = df[column].to_numpy(np.float64)
        except:
            result_dict[column] = df[column].to_numpy()
    return result_dict       
