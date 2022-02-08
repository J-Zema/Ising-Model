from __future__ import division
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from numba import jit

class Ising():
    ''' Simulating the Ising model '''

    ## monte carlo moves
    @staticmethod
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
                elif rand() < np.exp(-cost * beta):
                    s *= -1
                config[a, b] = s
        return config

    def simulate(self):
        ''' This module simulates the Ising model'''
        N, temp = 10, 1  # Initialse the lattice
        config = 2 * np.random.randint(2, size=(N, N)) - 1
        f = plt.figure(figsize=(15, 15), dpi=100)
        self.configPlot(f, config, 0, N, 1)

        msrmnt = 1800 + 1
        for i in range(msrmnt):
            self.mcmove(config, N, 1.0 / temp)
            if i == 10:       self.configPlot(f, config, i, N, 2)
            elif i == 100:       self.configPlot(f, config, i, N, 3)
            elif i == 550:      self.configPlot(f, config, i, N, 4)
            elif i == 950:     self.configPlot(f, config, i, N, 5)
            elif i == 1600:    self.configPlot(f, config, i, N, 6)

    def configPlot(self, f, config, i, N, n_):
        ''' This modules plts the configuration once passed to it along with time etc '''

        sp = f.add_subplot(3, 3, n_)
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)
        plt.imshow(config)
        plt.title('Time=%d' % i)
        plt.axis('tight')


    plt.show()

rm = Ising()
rm.simulate()
plt.show()