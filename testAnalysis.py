import numpy as np
import pandas as pd


data = pd.read_csv("./testResult.csv")
result = [0, 0]
for index in range(1960):
    arr = np.array((data.iloc[index, 1:]))
    tmp = []
    for i in range(2):
        tmp.append(arr[2+i*15])
    if tmp[0] < tmp[1]:
        result[0] += 1
    elif tmp[0] > tmp[1]:
        result[1] += 1
print(result)
