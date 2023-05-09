import pandas as pd
import numpy as np
from io import StringIO

def get_train ():
    with open("train.csv", "r") as f:
        train_data = f.read()
    train_data = pd.read_csv(StringIO(train_data))

    #used_entry = ["Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Views", "Likes", "Stream", "Comments"]

    used_entry = ["Key", "Loudness", "Tempo", "Instrumentalness"]

    train_x = [[] for _ in range(len(train_data))]
    for entry in used_entry:
        for i in range(len(train_data)):
            train_x[i].append(train_data[entry][i])
    train_y = [_ for _ in train_data['Danceability']]

    return (train_y, train_x)

def get_test ():
    with open("test.csv", "r") as f:
        test_data = f.read()
    test_data = pd.read_csv(StringIO(test_data))

    #used_entry = ["Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms", "Views", "Likes", "Stream", "Comments"]

    used_entry = ["Key", "Loudness", "Tempo", "Instrumentalness"]

    test_x = [[] for _ in range(len(test_data))]
    for entry in used_entry:
        for i in range(len(test_data)):
            test_x[i].append(test_data[entry][i])

    return test_x