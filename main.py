import numpy as np
import sympy as sym
from sympy.abc import x, y
import pandas as pd
import matplotlib.pyplot as plt
import time


class Input:
    data = pd.read_csv("./testData.csv")

    @staticmethod
    def userInput():
        n = int(input("좌표의 개수를 입력: "))
        coorList = []
        for i in range(n):
            coorList.append(list(map(int, input("x좌표와 y좌표를 순서대로 공백으로 구분하여 입력: ").split())))
        return n, np.array(coorList)

    @staticmethod
    def testInput(index):
        arr = np.array((Input.data.iloc[index, 1:]).dropna())
        arr = np.reshape(arr, (-1, 2))
        return len(arr), arr


class HalfwayAlgorithm:
    @staticmethod
    def arithmeticMean(coorList):           ##### Method 1 #####
        return np.mean(coorList, axis=0)

    @staticmethod
    def __removeOutlier(n, nList, zScore=1.282):  # by 80% confidence interval (zScore = 1.282)
        if n < 3:
            return nList
        sqMean = sum(nList ** 2) / n
        mean = sum(nList) / n
        sd = (sqMean - mean ** 2) ** (1 / 2)
        se = sd / n ** (1 / 2)
        corrVal = 0
        for i in range(n):
            if nList[i - corrVal] < mean - zScore * se or nList[i - corrVal] > mean + zScore * se:
                np.delete(nList, i - corrVal, 0)
                corrVal += 1
        return nList

    @staticmethod
    def trimmedMean(n, coorList):           ##### Method 2 #####
        if n < 3:
            return HalfwayAlgorithm.arithmeticMean(coorList)
        xList = HalfwayAlgorithm.__removeOutlier(n, coorList[:, 0])
        yList = HalfwayAlgorithm.__removeOutlier(n, coorList[:, 1])
        return np.array([np.mean(xList), np.mean(yList)])

    @staticmethod
    def __tmpFunc(point, n, coorList):
        return (np.tile(point, (n, 1)) - coorList) ** 2

    @staticmethod
    def lenSum(point, n, coorList):  # 모든 길이의 합 함수(f)
        tmp = HalfwayAlgorithm.__tmpFunc(point, n, coorList)
        return np.sum(np.sum(tmp, axis=1) ** (1 / 2))

    @staticmethod
    def dfdx(point, n, coorList):
        tmp = HalfwayAlgorithm.__tmpFunc(point, n, coorList)
        tmp = np.sum(tmp, axis=1) ** (1 / 2)
        return np.sum(np.repeat(point[0], n) - coorList[:, 0] / tmp)

    @staticmethod
    def dfdy(point, n, coorList):
        tmp = HalfwayAlgorithm.__tmpFunc(point, n, coorList)
        tmp = np.sum(tmp, axis=1) ** (1 / 2)
        return np.sum(np.repeat(point[1], n) - coorList[:, 1] / tmp)

    @staticmethod
    def sd(point, n, coorList):  # 모든 길이의 분산 함수(제평-평제)(g)
        return (np.sum(HalfwayAlgorithm.__tmpFunc(point, n, coorList)) / n - (
                    HalfwayAlgorithm.lenSum(point, n, coorList) / n) ** 2)**(1/2)

    @staticmethod
    def gradientdescent(n, coorList):       ##### Method 3 #####
        lr = 250  # 학습률
        eps = 0.01  # 종료조건
        val = HalfwayAlgorithm.arithmeticMean(coorList)
        func = 0
        for p in coorList:
            func += (((x - p[0]) ** 2) + ((y - p[1]) ** 2)) ** (1 / 2)
        pdx = sym.diff(func, x)
        pdy = sym.diff(func, y)
        grad = np.array([pdx.subs([(x, val[0]), (y, val[1])]), pdy.subs([(x, val[0]), (y, val[1])])])
        count = 0
        while ((np.sum(grad ** 2)) ** (1 / 2) > eps) and count < 250:
            val = val - lr * grad
            grad = np.array([pdx.subs([(x, val[0]), (y, val[1])]), pdy.subs([(x, val[0]), (y, val[1])])])
            count += 1
        if count >= 250:
            print("###### Gradient Descent Fail ######")
        return val

    @staticmethod
    def convexhull(n, coorList):
        tmpList = list(coorList[:])
        if n < 4:
            return tmpList, []
        xMin = xMax = tmpList[0][0]
        for i in range(1, n):
            if tmpList[i][0] < xMin:
                xMin = tmpList[i][0]
            elif tmpList[i][0] > xMax:
                xMax = tmpList[i][0]
        if xMin == xMax:
            return sorted(tmpList, key=lambda x: x[1])
        minArr, maxArr = 0, n - 1
        for j in range(n):
            if tmpList[j][0] == xMin:
                tmpList[j], tmpList[minArr] = tmpList[minArr], tmpList[j]
                minArr += 1
        for k in range(n - 1, minArr - 1, -1):
            if tmpList[k][0] == xMax:
                tmpList[k], tmpList[maxArr] = tmpList[maxArr], tmpList[k]
                maxArr -= 1
        minList = sorted(tmpList[0:minArr], key=lambda x: x[1])
        maxList = sorted(tmpList[maxArr + 1:n], key=lambda x: x[1], reverse=True)
        midList = tmpList[minArr:maxArr + 1] + maxList[0:1]
        chList = minList[:]

        for a in range(2):
            while True:
                midLen = len(midList)
                chLen = len(chList)
                candidateIdx = midLen - 1
                m = (midList[candidateIdx][1] - chList[chLen - 1][1]) / (
                        midList[candidateIdx][0] - chList[chLen - 1][0])
                for i in range(midLen - 1):
                    dx = midList[i][0] - chList[chLen - 1][0]
                    if (a == 0 and dx > 0) or (a == 1 and dx < 0):
                        dy = midList[i][1] - chList[chLen - 1][1]
                        if m < dy / dx:
                            m = dy / dx
                            candidateIdx = i
                        elif m == dy / dx:
                            if midList[candidateIdx][0] > midList[i][0]:
                                candidateIdx = i
                if candidateIdx != midLen - 1:
                    chList.append(midList[candidateIdx])
                    midList.pop(candidateIdx)
                else:
                    break
            midList.pop()
            if a == 0:
                chList += maxList
                midList.append(minList[0])

        return chList, midList

    @staticmethod
    def __centroid(n, stdList):
        if n <= 3:
            return list(HalfwayAlgorithm.arithmeticMean(stdList))
        cx, cy, a = 0, 0, 0
        for i in range(n):
            tmp = stdList[i][0] * stdList[(i + 1) % n][1] - stdList[(i + 1) % n][0] * stdList[i][1]
            a += tmp
            cx += (stdList[i][0] + stdList[(i + 1) % n][0]) * tmp
            cy += (stdList[i][1] + stdList[(i + 1) % n][1]) * tmp
        if a == 0:
            return list(HalfwayAlgorithm.arithmeticMean(stdList))
        else:
            return [cx / (3 * a), cy / (3 * a)]

    @staticmethod
    def centroidMean(n, coorList):          ##### Method 4 #####
        if n < 4:
            return HalfwayAlgorithm.arithmeticMean(coorList)
        xResult, yResult = 0, 0
        candidateList = []  # 무게중심x, 무게중심y, 가중값
        tmp = coorList[:]
        while True:
            ch, inner = HalfwayAlgorithm.convexhull(len(tmp), tmp)
            t = HalfwayAlgorithm.__centroid(len(ch), ch)
            t.append(len(ch))
            candidateList.append(t)
            if inner:
                tmp = inner
            else:
                break
        divisor = 0
        for i in range(len(candidateList)):
            xResult += candidateList[i][0] * candidateList[i][2]
            yResult += candidateList[i][1] * candidateList[i][2]
            divisor += candidateList[i][2]
        return np.array([xResult / divisor, yResult / divisor])


# single test
n, cl = Input.testInput(1000)                  ## 0 - 1959 사이의 값 parameter로 입력
clt = np.transpose(cl)
plt.scatter(clt[0], clt[1], s=5, c='black')

a1 = HalfwayAlgorithm.arithmeticMean(cl)
print(a1)
print(HalfwayAlgorithm.lenSum(a1, n, cl))
print(HalfwayAlgorithm.sd(a1, n, cl))
plt.scatter(a1[0], a1[1], s=5)
print("======"*5)

a2 = HalfwayAlgorithm.trimmedMean(n, cl)
print(a2)
print(HalfwayAlgorithm.lenSum(a2, n, cl))
print(HalfwayAlgorithm.sd(a2, n, cl))
plt.scatter(a2[0], a2[1], s=5, c='red')
print("======"*5)

a3 = HalfwayAlgorithm.gradientdescent(n, cl)
print(a3)
print(HalfwayAlgorithm.lenSum(a3, n, cl))
print(HalfwayAlgorithm.sd(a3, n, cl))
plt.scatter(a3[0], a3[1], s=5, c='green')
print("======"*5)

a4 = HalfwayAlgorithm.centroidMean(n, cl)
print(a4)
print(HalfwayAlgorithm.lenSum(a4, n, cl))
print(HalfwayAlgorithm.sd(a4, n, cl))
plt.scatter(a4[0], a4[1], s=5, c='blue')
print("======"*5)

plt.show()







# # Test code
# result = []
# for i in range(1960):           #1960개의 test case
#     print(i, "case start")
#     n, cl = Input.testInput(i)
#     a, rt = [], []
#     start = time.time()
#     a1 = HalfwayAlgorithm.arithmeticMean(cl)
#     rt.append(time.time() - start)
#     a.append(list(a1))
#
#     start = time.time()
#     a2 = HalfwayAlgorithm.trimmedMean(n, cl)
#     rt.append(time.time() - start)
#     a.append(list(a2))
#
#     start = time.time()
#     a3 = HalfwayAlgorithm.gradientdescent(n, cl)
#     rt.append(time.time() - start)
#     a.append(list(a3))
#
#     start = time.time()
#     a4 = HalfwayAlgorithm.centroidMean(n, cl)
#     rt.append(time.time() - start)
#     a.append(list(a4))
#
#     tmp = []
#     for j in range(4):
#         tmp.extend(a[j])
#         tmp.append(HalfwayAlgorithm.lenSum(a[j], n, cl))
#         tmp.append(HalfwayAlgorithm.sd(a[j], n, cl))
#         tmp.append(rt[j])
#     result.append(tmp)
#     print(i, "case complete,", sum(rt), "taken")
#
# col1 = ['a1 x', 'a1 y', 'a1 lenSum', 'a1 sd', 'a1 rt']
# col2 = ['a2 x', 'a2 y', 'a2 lenSum', 'a2 sd', 'a2 rt']
# col3 = ['a3 x', 'a3 y', 'a3 lenSum', 'a3 sd', 'a3 rt']
# col4 = ['a4 x', 'a4 y', 'a4 lenSum', 'a4 sd', 'a4 rt']
#
# output = pd.DataFrame(result, columns=col1+col2+col3+col4)
#
# output.to_csv('testResult.csv')


