#!/usr/bin/env python3
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

def get_alphas(ts):
    ts = ts[1:]
    n = len(ts)

    A = []
    for i in range(n):
        tmp = []
        for j in range(n):
            if i <= j:
                tmp.append((1-ts[j]) - (1-ts[i]) * (1 - ts[j]))
            else:
                tmp.append((1-ts[i]) - (1-ts[i]) * (1 - ts[j]))
        A.append(tmp)
    A = np.array(A)

    B = []
    for i in range(n):
        tmp0 = integrate.quad(lambda x: x * f(x) - ts[i] * f(x), ts[i], 1)[0]
        tmp1 = integrate.quad(lambda x: x * f(x), 0, 1)[0]
        B.append(tmp0 - (1-ts[i]) * tmp1)
    B = np.array(B)

    X = np.linalg.solve(A, B)
    alpha0 = -sum(X[i] * (1-ts[i]) for i in range(n))
    alpha0 = np.array([alpha0])

    alphas = np.concatenate((alpha0, X))
    return alphas


def relu_fem(ts, alphas, x):
    n = len(ts)
    out = 0
    for i in range(n):
        out += alphas[i] * np.maximum(x - ts[i], 0)
    return out

def e0(x, ts, alphas):
    n = len(ts)
    out = 0
    for i in range(n):
        if x-ts[i] > 0:
            out += alphas[i]
    return out**2

def e1(x, ts, alphas):
    n = len(ts)
    out = 0
    for i in range(n):
        if x-ts[i] > 0:
            out += alphas[i] * (x-ts[i])
    return f(x) * out

        

def energy_norm(ts, alphas):
    #e0 =  (1/2)*integrate.quad(lambda x: sum(alphas * ((x - ts)>0))**2, 0, 1, limit=100)[0]
    e_0 =  (1/2)*integrate.quad(e0, 0, 1, limit=100, args=(ts,alphas))[0]
    #e1 = integrate.quad(lambda x: f(x) * sum(alphas * np.maximum(x - ts, 0)), 0, 1, limit=100)[0]
    e_1 = integrate.quad(e1, 0, 1, limit=100, args=(ts,alphas))[0]
    return e_0 - e_1

def f(x):
    return np.exp(x)


nodes = 2

# initalise ts as evenly spaced nodes
ts = np.linspace(0 , 1, nodes + 1)[:nodes]
print(0, ts)
alphas = get_alphas(ts)

points = np.linspace(0.,1., 1000)
plt.plot(points, relu_fem(ts, alphas, points))
plt.plot(points, np.e*points - points - np.exp(points) + 1)
#plt.plot(points, 1/6*(points - points**3))

for i in range(1):
    gradient = approx_fprime(ts, energy_norm, 1e-2, alphas)
    print('grad:', gradient)
    ts = ts - 0.1 * gradient
    print(i+1, ts)
    ts[0] = 0.
    alphas = get_alphas(ts)
plt.plot(points, relu_fem(ts, alphas, points), color='r')

plt.grid()
plt.savefig('test.svg')
