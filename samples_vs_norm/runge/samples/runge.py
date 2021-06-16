#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
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

    def get_2_norm(self):
        vals = np.linspace(-1, 1, 1000000)
        runge_vals = 1 / (1 + 25 * (vals**2))
        interp_vals = self.interpolate(vals)
        return np.linalg.norm(interp_vals - runge_vals, 2)


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
        y = torch.true_divide(1, 1 + 25 * (x**2))
        return x, y

    def checkpoint(self):
        self.model_out_path = "models/{:03d}_neurons{:03d}_samples.pth".format(self.p, self.n)
        torch.save(self.model, self.model_out_path)

    def get_2_norm(self):
        vals = torch.linspace(-1, 1, 1000000).reshape(1000000, 1)
        model_vals = self.model(vals).detach()
        runge_vals = torch.true_divide(1, 1 + 25 * (vals**2))
        return np.linalg.norm(model_vals - runge_vals, 2)

    def train_to_epochs(self, max_epoch):
        x, y = self.get_samples()

        self.smallest_loss = 1000
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
    ax.set_ylim([-0.1, 1.1])

    # plot runge function
    points = np.linspace(-1.25, 1.25, 1000)
    runge_points = 1 / (1 + 25 * (points**2))
    ax.plot(points, runge_points, '-', label='Runge function')


    # plot l-interpolation and get 2 norm
    samps = np.linspace(-1, 1, samples)
    runge_samps = 1 / (1 + 25 * (samps**2))
    l_2norm = 0
    if samples <= 5:
        l = LagrangePoly(samps, runge_samps)
        l_points = l.interpolate(points)
        l_2norm = l.get_2_norm()
        ax.plot(points, l_points, '--', label='Lagrange interpolation')

    # train and plot nn model and get 2 norm
    points = torch.linspace(-1.25, 1.25, 1000).reshape(1000, 1)
    m = nSample_pNeuronNetwork(samples, neurons)
    m.train_to_epochs(epochs)
    m_2norm = m.get_2_norm()
    m_points = m.model(points).detach()
    ax.plot(points, m_points, '-.', label='NN approxmation')
    ax.plot(samps, runge_samps, '.', label='Samples')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, 'both')
    legend = ax.legend(loc='upper left')
    fig.savefig('graphs/runge_{:03d}neurons_{:03d}samples.pgf'.format(neurons, samples))

    return [l_2norm, m_2norm, m.smallest_loss.detach(), m.epochs]

print('Samples & Lagrange $\\varepsilon_{rel}$ & Neural network $\\varepsilon_{rel}$ & Loss & Iterations \\\\')
for i in [3, 5, 7, 9, 13, 17, 21, 25, 33, 41, 49]:
    out = run_test(6, i, 500000)
    print('{} & {:.6g} & {:.6g} & {:.6g} & {} \\\\'.format(i, *out))
