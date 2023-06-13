import matplotlib.pyplot as plt
import pandas as pd

Eval = pd.read_csv("score.csv").iloc[:,0]
Public = pd.read_csv("score.csv").iloc[:,1]
plt.scatter(Eval, Public)
plt.savefig("Eval.png", dpi=300)
