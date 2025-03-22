import numpy as np
import matplotlib.pyplot as plt


def plotting_1d(y, llabel, xlabel, ylabel, title, save_npy=False, save_figure=False, x=None):
    ys = y
    if not isinstance(ys, np.ndarray):
        ys = y.detach().numpy()

    fig, ax = plt.subplots()
    if x is not None:
        l1, = ax.plot(x, ys, 'r', label=llabel)
    else:
        l1, = ax.plot(ys, 'r', label=llabel)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(handles=[l1], loc=1)
    if save_figure:
        plt.savefig(title + '.png', dpi=300)
    else:
        plt.show()
    if save_npy:
        np.save(title, ys)


def plotting_simple_2d(value):
    for idx in range(value.shape[-1]):
        fig, ax = plt.subplots()
        l1, = ax.plot(value[:, idx], 'r', label='cam')
        plt.show()

def plotting_simple_scatter_2d(value, x):
    for idx in range(value.shape[-1]):
        fig, ax = plt.subplots()
        ax.scatter(x, value[:, idx], s=np.ones_like(x) * 0.1)
        ax.set_ylim([0, 0.4])
        plt.show(block=False)


def plotting_scatter_2d(y, llabel, xlabel, ylabel, title, save_npy=False, save_figure=False, x=None):
    ys = y
    if not isinstance(ys, np.ndarray):
        ys = y.detach().numpy()

    if not isinstance(x, np.ndarray):
        x = x.detach().numpy()

    x = x / np.pi * 180.0

    fig, ax = plt.subplots(1, ys.shape[-1], sharex=True, sharey=True)
    fig.set_size_inches((12, ys.shape[-1]))

    for i in range(ys.shape[-1]):
        ax[i].set_title(title + '{0}'.format(i), fontsize=20)
        ax[i].set_xlabel(xlabel, fontsize=20)
        ax[i].set_ylabel(ylabel, fontsize=20)
        ax[i].scatter(x, ys[:, i])
        ax[i].set_ylim([0, 30])
        ax[i].set_xlim([0, 60])

    if save_figure:
        plt.savefig(title + '.png', dpi=300)
    else:
        plt.show()
    if save_npy:
        np.save(title, ys)