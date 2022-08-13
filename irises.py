import numpy as np
import pandas as pd
from scipy import stats
df=pd.read_csv("data1.csv")
irises = np.array(df[["SepalLength","SepalWidth",'PetalLength','PetalWidth']])
types =np.array (df[["Name"]]).reshape(150)
new_irises = np.load('new_irises.npy')
n, m = len(irises), len(new_irises)
def calc_one_loop(new_points, points):
    m, n = len(new_points), len(points)    
    d = np.zeros((m, n))
    for i in range(m):
        d[i] = np.square(new_points[i]-points).sum(axis=1)
    return d
k = 10
d = calc_one_loop(new_irises, irises)
k_nearest = np.argpartition(d,k,axis=1)[: ,: k]
k_nearest_types = types[k_nearest]
predicted_types = stats.mode(k_nearest_types, axis=1).mode.reshape(m)
print(predicted_types)