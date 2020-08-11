inputs = [1, 7, 5]
weights = [0.8, 0.1, 0]


def sum(inp, w):
    s = 0
    for i in range(3):
        s += inp[i] * w[i]
    return s


def stepFunction(s):
    if (s >= 1):
        return 1
    return 0


s = sum(inputs, weights)
print("s=", s)

r = stepFunction(s)
print("r=", r)
