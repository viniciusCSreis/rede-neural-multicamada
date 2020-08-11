
import numpy as np

inputs = np.array([1, 7, 5])
weights = np.array([0.8, 0.1, 0])


def sum(inp, w):
    return inp.dot(w)


def stepFunction(s):
    if (s >= 1):
        return 1
    return 0


s = sum(inputs, weights)
print("s=", s)

r = stepFunction(s)
print("r=", r)
