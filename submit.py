from time import time
import subprocess
import matplotlib.pyplot as plt

def submit (y):
    subprocess.run('mkdir prediction', shell=True)
    subprocess.run('cp main.py prediction/', shell=True)
    subprocess.run('touch prediction/score', shell=True)
    filename = f'predict_{time()}'

    with open(f"prediction/{filename}.csv", "w") as f:
        print("id,Danceability", file = f)
        for i in range(len(y)):
            v = y[i]
            v = round(v)
            v = max(v, 0)
            v = min(v, 9)
            v = float(v)
            print(f"{17170 + i},{v}", file = f)

    bins = [i-0.5 for i in range(11)]
    plt.xticks(range(10))
    plt.hist(y, bins=bins, align='mid')
    plt.xlabel('Danceability')
    plt.ylabel('Count')
    plt.title(f'{filename} Histogram')
    plt.savefig(f'prediction/{filename}-hist.png', dpi=300)
