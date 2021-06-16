#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.utils.data as Data
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
matplotlib.rcParams['axes.unicode_minus'] = False

class LagrangePoly:

    def __init__(self, X, Y):
        self.n = len(X)
        self.X = np.array(X)
        self.Y = np.array(Y)

    def basis(self, x, j):
        b = [(x - self.X[m]) / (self.X[j] - self.X[m])
             for m in range(self.n) if m != j]
        return np.prod(b, axis=0) * self.Y[j]

    def interpolate(self, x):
        b = [self.basis(x, j) for j in range(self.n)]
        return np.sum(b, axis=0)

    def get_inf_norm(self):
        vals = np.linspace(-1, 1, 1000000)
        sin_vals = np.sin(vals * pi)
        interp_vals = self.interpolate(vals)
        return np.linalg.norm(interp_vals - sin_vals, np.inf)


class nSample_pNeuronNetwork:

    def __init__(self, n, p):
        self.n = n
        self.p = p
        torch.manual_seed(0)

        self.model = nn.Sequential(
                nn.Linear(1, p),
                nn.GELU(),
                nn.Linear(p, 1),
                )

        self.loss_fn = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-2)

    def get_samples(self):
        x = torch.linspace(-1, 1, self.n).reshape(self.n, 1)
        y = torch.sin(pi * x)
        return x, y

    def checkpoint(self):
        self.model_out_path = "models/{:03d}_neurons{:03d}_samples.pth".format(self.p, self.n)
        torch.save(self.model, self.model_out_path)

    def get_2_norm(self):
        vals = torch.linspace(-1, 1, 1000000).reshape(1000000, 1)
        model_vals = self.model(vals).detach()
        sin_vals = torch.sin(pi * vals)
        return np.linalg.norm(model_vals - sin_vals, 2)

    def train_to_epochs(self, max_epoch):
        x, y = self.get_samples()

        self.smallest_loss = 9
        epochs = 0
        while epochs < max_epoch and self.smallest_loss >= 5e-7:

            epochs += 1

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if loss < self.smallest_loss:
                self.smallest_loss = loss
                self.checkpoint()
                #print(loss.detach())

        self.epochs = epochs
        self.model = torch.load(self.model_out_path)


def run_test(neurons, samples, epochs):
    # setup plot
    plt.cla()
    fig, ax = plt.subplots()
    #fig.suptitle('\n'.join(wrap('Approximating sin(pi*x) using a neural network with {} neurons trained with {} evenly spaced points to {} epochs'.format(neurons, samples, epochs), 60)))
    ax.set_ylim([-1.1, 1.1])

    # plot sin(pi x)
    points = np.linspace(-1.25, 1.25, 1000)
    sin_points = np.sin(points * pi)
    ax.plot(points, sin_points, '-', label='sin(pi*x)')


    # plot l-interpolation and get inf norm
    samps = np.linspace(-1, 1, samples)
    sin_samps = np.sin(samps * pi)

    # train and plot nn model and get inf norm
    points = torch.linspace(-1.25, 1.25, 1000).reshape(1000, 1)
    m = nSample_pNeuronNetwork(samples, neurons)
    m.train_to_epochs(epochs)
    m_2_norm = m.get_2_norm()
    m_points = m.model(points).detach()
    ax.plot(points, m_points, '-.', label='NN Approximation')
    ax.plot(samps, sin_samps, '.', label='Samples')
    legend = ax.legend(loc='upper left')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, 'both')
    fig.savefig('graphs/sin_{:03d}neurons_{:03d}samples.pgf'.format(neurons, samples))

    return [m_2_norm, m.smallest_loss.detach(), m.epochs]

print('Neurons & Neural network $\\varepsilon_{rel} &  Loss & Iterations \\\\')
for i in [2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48]:
    out = run_test(i, 11, 500000)
    print('{} & {:.6g} & {:.6g} & {} \\\\'.format(i, *out))
