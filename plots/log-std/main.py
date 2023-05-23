from parse_data import get_train, get_test
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import pickle

# "Danceability","Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"
used_entry = ["Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"]
target_entry = ["Energy","Key","Loudness","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Duration_ms","Views","Likes","Stream","Album_type","Licensed","official_video","id","Track","Album","Uri","Url_spotify","Url_youtube","Comments","Description","Title","Channel","Composer","Artist"]

# Gen input data pickle
"""
y, x = get_train(used_entry)
x = np.array(x)
y = np.array(y)

pickle_name = 'pickles/train_data.pickle'
with open(pickle_name, 'wb') as f:
    pickle.dump(y, f)
    pickle.dump(x, f)
"""


# Read input data pickle
pickle_name = 'pickles/train_data.pickle'
with open(pickle_name, 'rb') as f:
    y = pickle.load(f)
    x = pickle.load(f)


max_labels = 10
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(max_labels))

x = x.transpose()
progress = tqdm(total = len(target_entry))
skip_entries = "\n-------------------------------------\n"

for i, entry in enumerate(used_entry):
    if entry not in target_entry:
        continue
    try:
        ny, nx = [], []
        for yi, xi in zip(y, x[i].astype(float)):
            if not np.isnan(xi):
                ny.append(yi)
                nx.append(xi)
        ny = np.array(ny)
        nx = np.array(nx)
        
        if entry == "Loudness":
            nx *= -1
        nx = np.maximum(nx, 0.00000000001)
        nx = np.log(nx)
        nx = scale(nx)
        plt.scatter(nx, ny, marker='o', s=3, alpha=0.2)  # Scatter plot with circular markers
    except:
        skip_entries += f"[WARN] Cannot convert \033[91m{entry}\033[0m to float, skipping\n"
        progress.update(1)
        continue

    plt.xlabel(f'{entry}')  # Add label to x-axis
    plt.ylabel('Danceability')  # Add label to y-axis
    plt.title('Log Scatter Plot')  # Add title to the graph
    plt.savefig(f'plots/{entry}.png', dpi=300)  # Save the plot as a PNG file
    plt.clf()
    progress.update(1)

print(skip_entries)