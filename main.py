import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys 

filename = sys.argv[1]
prediction_name = filename.split('/')[2][:-4]

df = pd.read_csv(filename)
ids = df['id'].values 
y = df['Danceability'].values

bins = [i-0.5 for i in range(11)]
plt.xticks(range(10))
plt.hist(y, bins=bins, align='mid')
plt.xlabel('Danceability')
plt.ylabel('Count')
plt.title(f'{prediction_name} Histogram')

plt.savefig(f'{filename[:-4]}-hist.png', dpi=300)