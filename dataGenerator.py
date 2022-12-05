# 0 ~ 50000까지의 값을 가지는 m개의 (x, y)쌍을 20개 생성
# m = range(3,101) n<= 1000
import random
import numpy as np
import pandas as pd
# n = int(input("Number of test case: "))
# mini = int(input("Minimum number of points: "))
# maxi = int(input("Maximum number of points: "))

output = []
for m in range(3, 101):
    interim = []
    for _ in range(20):
        tmp = []
        for _ in range(m):
            tmp.append(random.randrange(0, 50001))
            tmp.append(random.randrange(0, 50001))
        interim.append(tmp)
    output.extend(interim)
output = pd.DataFrame(output)

output.to_csv('testData.csv')