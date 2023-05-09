from time import time

def submit (y):
    with open(f"predict_{time()}.csv", "w") as f:
        print("id,Danceability", file = f)
        for i in range(len(y)):
            v = y[i]
            v = round(v)
            v = max(v, 0)
            v = min(v, 9)
            v = float(v)
            print(f"{17170 + i},{v}", file = f)
