
import numpy as np

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
outputs = np.array([0, 0, 0, 1])
weights = np.array([0.0, 0.0])
learnRate = 0.1


def stepFunction(s):
    if (s >= 1):
        return 1
    return 0


def calc(inp):
    s = inp.dot(weights)
    print("s=", s)
    return stepFunction(s)


def train():
    sumErr = 1
    while sumErr != 0:
        sumErr = 0
        for i in range(len(outputs)):
            r = calc(inputs[i])
            err = abs(outputs[i] - r)
            sumErr += err
            for j in range(len(weights)):
                weights[j] = weights[j] + (learnRate * inputs[i][j] * err)
        print("new weight =", weights[j])


train()
print("final weights:", weights)
