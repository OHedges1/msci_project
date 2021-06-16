#!/usr/bin/env python3
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

def get_alphas(ts):
    n = len(ts)

    A = []
    for i in range(1, n):
        tmp = []

        for j in range(1, n):
            if i <= j:
                tmp.append((1-ts[j]) - (1-ts[i]) * (1 - ts[j]) / (1 - ts[0]))
            else:
                tmp.append((1-ts[i]) - (1-ts[i]) * (1 - ts[j]) / (1 - ts[0]))
        A.append(tmp)

    A = np.array(A)

    B = []
    for i in range(1, n):
        tmp0 = integrate.quad(lambda x: x * f(x) - ts[i] * f(x), ts[i], 1)[0]
        tmp1 = integrate.quad(lambda x: x * f(x) - ts[0] * f(x), ts[0], 1)[0]
        B.append(tmp0 - (1 - ts[i]) * tmp1 / (1 - ts[0]))
    B = np.array(B)

    X = np.linalg.solve(A, B)

    alpha0 = -sum(X[i] * (1-ts[i+1]) for i in range(n-1)) / (1 - ts[0])
    alpha0 = np.array([alpha0])
    alphas = np.concatenate((alpha0, X))

    return alphas



def relu_fem(ts, alphas, x):
    n = len(ts)
    out = 0
    for i in range(n):
        out += alphas[i] * np.maximum(x - ts[i], 0)
    return out


def energy_norm(alphas, ts):
    e0 = (1/2) *  integrate.quad(lambda x: sum(alphas * ((x - ts)>0))**2, 0, 1, limit=200)[0]
    e1 = integrate.quad(lambda x: f(x) * sum(alphas * np.maximum(x - ts, 0)), 0, 1, limit=100)[0]
    return e0 - e1 


def f(x):
    return np.exp(x)


nodes = 4

# initalise ts as evenly spaced nodes
ts = np.linspace(.0 , 1, nodes + 1)[:nodes]
alphas = get_alphas(ts)
print(ts)

points = np.linspace(0.,1., 1000)
plt.plot(points, relu_fem(ts, alphas, points))
plt.plot(points, np.e*points - points - np.exp(points) +1)
#plt.plot(points, 1/6*(points - points**3))

for i in range(10):
    gradient = approx_fprime(ts, energy_norm, 1e-8, alphas)
    ts = ts - 0.1 * gradient
    print(i, ts)
    #ts[0] = 0.
    alphas = get_alphas(ts)
plt.plot(points, relu_fem(ts, alphas, points), color='r')

plt.savefig('test.svg')
