import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoidDerivative(r):
    return r * (1 - r)


inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

outputs = np.array([
    [0],
    [1],
    [1],
    [0],
])

w0 = np.array([
    [-0.424, -0.740, -0.961],
    [0.358, -0.577, -0.469],
])
w0 = 2*np.random.random((2, 3)) - 1

w1 = np.array([
    [-0.017],
    [-0.893],
    [0.148],
])
w1 = 2*np.random.random((3, 1)) - 1

trainingTime = 1000
trainingTax = 10
moment = 1

for i in range(trainingTime):
    inp = inputs
    s0 = np.dot(inp, w0)
    r0 = sigmoid(s0)
    s1 = np.dot(r0, w1)
    r1 = sigmoid(s1)

    errR1 = outputs - r1
    errAvg = np.mean(np.abs(errR1))
    delta1 = errR1 * sigmoidDerivative(r1)
    w1T = w1.T
    deltaW1 = delta1.dot(w1T)
    delta0 = deltaW1 * sigmoidDerivative(r0)

    newW1 = r0.T.dot(delta1)
    w1 = (w1 * moment) + (newW1 * trainingTax)

    newW0 = inp.T.dot(delta0)
    w0 = (w0 * moment) + (newW0 * trainingTax)
    print("avg", errAvg)

print("w0'", w0)
print("w1'", w1)
