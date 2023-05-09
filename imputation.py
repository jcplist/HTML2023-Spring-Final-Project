import numpy as np
from sklearn.impute import SimpleImputer

def mean_imputation (x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    return imp.transform(x)