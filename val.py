import pandas as pd
a = pd.read_csv('val.csv')
b = pd.read_csv('t.csv')

err = 0
index = 0
for i in range(len(a)):
    id = a.loc[i]["id"]
    while b.loc[index]["id"] < id:
        index += 1
    err += abs(a.loc[i]["Danceability"] - b.loc[index]["Danceability"])

print(err / len(a))