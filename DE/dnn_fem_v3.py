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


def f(x):
    return np.exp(x)

def get_grad(alphas, ts):
    ts = np.append(ts, [1.])
    n = len(alphas)
    G = 0 
    for i in range(1, n):
        G += alphas[i] * (ts[i] - 1)

    grad = []
    for p in range(1, n):
        
        E1 = 0.5 * ((G + sum(alphas[1: p])) ** 2 - (G + sum(alphas[1:p+1])) ** 2)
        for i in range(1, n+1):
            E1 += alphas[p] * (ts[i] - ts[i-1]) * (G + sum(alphas[1:i]))

        E2 = alphas[p] * (1 - np.exp(ts[p+1]) + np.exp(ts[p]))
        for i in range(p+1, n):
            E2 -= alphas[p] * (np.exp(ts[i+1]) - np.exp(ts[i]))
        #print('E1: {}, E2: {}'.format(E1, E2))

        grad.append(E1 - E2)
    return np.array(grad)

nodes = 10

# initalise ts as evenly spaced nodes
ts = np.linspace(0 , 1, nodes + 1)[:nodes]
print(0, ts)
alphas = get_alphas(ts)

points = np.linspace(0.,1., 1000)
plt.plot(points, relu_fem(ts, alphas, points), label='old')
plt.plot(points, np.e*points - points - np.exp(points) + 1, label='solution')
#plt.plot(points, 1/6*(points - points**3))

for i in range(500):
    #print(i)
    #g_old = (1 - 2*ts[1]) * (alphas[1]**2)/2 - alphas[1] * (1 - np.e + np.e ** ts[1])
    gradient = get_grad(alphas, ts)
    #print('gradients: old {}, new {}'.format(g_old, gradient))
    #print('gradient: {}'.format(gradient))
    ts[1:] = ts[1:] - 0.5 * gradient
    #print('iteration: {}, ts: {}'.format(i+1, ts))
    alphas = get_alphas(ts)
plt.plot(points, relu_fem(ts, alphas, points), color='r', label='new')

print('gradient: {}'.format(gradient))
print('iteration: {}, ts: {}'.format(i+1, ts))

plt.grid()
plt.legend()
plt.savefig('test.svg')
