import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import animation


@jit(nopython=True)
def mcmove(config, N, beta):
    ''' This is to execute the monte carlo moves using
    Metropolis algorithm such that detailed
    balance condition is satisified'''
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = config[a, b]
            nb = config[(a + 1) % N, b] + config[a, (b + 1) % N] + config[(a - 1) % N, b] + config[a, (b - 1) % N]
            cost = 2 * s * nb
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * 1/beta):
                s *= -1
            config[a, b] = s
    return config

def simulate(N):
    ''' This module simulates the Ising model'''
    config = 2 * np.random.randint(2, size=(N, N)) - 1
    return config


N, temp = 10, 4 # Initialse the lattice
mc_steps=50 #MC steps
fps = 5

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )
config = simulate(N)
a = config
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)
plt.xticks([])
plt.yticks([])


def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )
    for j in range(i):
        val = mcmove(config, N, temp)
        im.set_array(val)
        plt.title("T = {}, {} MCS ".format(str(temp), str(i)))
    return [im]
#
anim = animation.FuncAnimation(
                               fig,
                               animate_func,
                               frames = mc_steps,
                               interval = 10000, # in ms
                               )
anim.save('animN={}_t={}.gif'.format(str(N), str(temp)), fps=fps, writer='Pillow' )
