import pandas as pd
X_val = pd.read_csv('test.csv')
y_val = pd.read_csv('tt.csv').iloc[:,1:]

print(X_val, y_val)

D_val = pd.concat([y_val, X_val], axis=1)
D_val.dropna(subset='Danceability').to_csv("val.csv", index=False)


