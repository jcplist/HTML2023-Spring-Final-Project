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
target_entry = ["Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"]

y, x = read_pickle()
x = x.transpose()
n = len(y)

progress = tqdm(total = len(target_entry))
skip_entries = "\n-------------------------------------\n"

for index, entry in enumerate(used_entry):
    if entry not in target_entry:
        continue
    try:
        cx = x[index].astype(float)
    except:
        skip_entries += f"[WARN] Cannot convert \033[91m{entry}\033[0m to float, skipping\n"
        progress.update(1)
        continue
    
    cy, cx, cnt = clean_nan(y, cx)
    print(f'{cnt} / {n} = {cnt/n*100:.2f}%')

    # Set upper-bound and lower-bound
    cx = np.maximum(cx, np.quantile(cx, 0.01))
    cx = np.minimum(cx, np.quantile(cx, 0.99))

    plt.hist(cx, bins=200)  # Scatter plot with circular markers
    plt.xlabel(f'{entry}')  # Add label to x-axis
    # plt.ylabel('Danceability')  # Add label to y-axis
    plt.title('Train Raw Histogram')  # Add title to the graph
    plt.savefig(f'plots/{entry}.png', dpi=300)  # Save the plot as a PNG file
    plt.clf()
    progress.update(1)

print(skip_entries)