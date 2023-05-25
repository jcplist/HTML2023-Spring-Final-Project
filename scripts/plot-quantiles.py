from parse_data import get_train, get_test
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import pickle

def clean_nan(y, x):
    clean_y, clean_x = [], []
    cnt = 0
    for yi, xi in zip(y, x):
        if np.isnan(xi):
            cnt += 1
        else:
            clean_y.append(yi)
            clean_x.append(xi)
    return np.array(clean_y), np.array(clean_x), cnt


pickle_name = 'pickles/train_data.pickle'
def gen_pickle():
    y, x = get_train(used_entry)
    x = np.array(x)
    y = np.array(y)
    with open(pickle_name, 'wb') as f:
        pickle.dump(y, f)
        pickle.dump(x, f)


def read_pickle():
    with open(pickle_name, 'rb') as f:
        y = pickle.load(f)
        x = pickle.load(f)
    return y, x


# "Danceability","Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"
used_entry = ["Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"]
target_entry = ["Views"]


y, x = read_pickle()
x = x.transpose()
n = len(y)

skip_entries = "\n-------------------------------------\n"

for index, entry in enumerate(used_entry):
    if entry not in target_entry:
        continue
    try:
        cx = x[index].astype(float)
    except:
        skip_entries += f"[WARN] Cannot convert \033[91m{entry}\033[0m to float, skipping\n"
        continue
    
    cy, cx, cnt = clean_nan(y, cx)
    print(f'{cnt} / {n} = {cnt/n*100:.2f}%')

    quantiles = []
    for i in range(101):
        qi = np.quantile(cx, i/100)
        quantiles.append(qi)
        print(f'Q{i} = {qi:.2f}')
    
    for top in range(90, 101):
        plt.plot([i for i in range(top+1)], quantiles[:top+1], linewidth=2.0)
        plt.savefig(f'plots/Q{top}.png', dpi=300)
        plt.clf()
    
    # for bot in range(0, 10):
    #     plt.plot([i for i in range(bot, 101)], quantiles[bot:], linewidth=2.0)
    #     plt.savefig(f'plots/Q{bot}.png', dpi=300)
    #     plt.clf()