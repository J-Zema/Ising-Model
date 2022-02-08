import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def initial_state(L):
    """Initializes spins configuration"""
    initialization = np.ones((L, L), dtype=int)
    return initialization

@jit(nopython=True)
def mc_move(config, T):
    """Monte Carlo move using Metropolis algorithm"""
    L = len(config)
    for i in range(L*L):
        x = np.random.randint(0, L)
        y = np.random.randint(0, L)
        spin = config[x, y] #wylosowany spin
        neighbors = config[(x+1) % L, y] + config[x, (y+1) % L] + config[(x-1) % L, y] + config[x, (y-1) % L]#cykliczna
        deltaE = 2*spin*neighbors  #2*wylosowany spin*wartość spinów sąsiadów
        if deltaE <= 0:
            spin *= -1
        elif np.random.rand() < np.exp(-deltaE/(T)):
            spin *= -1
        config[x, y] = spin
    return config

def a_spin_value(config):
    '''average spin value '''
    average = np.sum(config)
    return average/(len(config)*len(config))


@jit(nopython=True)
def calcMag(config):
    '''Magnetization of a given configuration'''
    N = len(config)
    mag = np.sum(config)
    return mag/(N*N)

# eqSteps = 30000
# mcSteps = 230000    #  number of MC steps
#eqSteps = 10**4
#mcSteps = 990000
eqSteps = 1024
mcSteps = 1024

dt = 32  #  number of temperature points
t = np.linspace(1, 3.5, dt) #temperature
S, M, = np.zeros(dt), np.zeros(dt),
L = [10, 50, 100]
n1, n2 = 1.0/(mcSteps), 1.0/(mcSteps*mcSteps)



def draw(magnetization=True):
    """Drawing magnetization or susceptibility"""
    for N in L:
        for tt in range(dt):
            S1 = M1 = 0
            iT = t[tt]
            config = initial_state(N)
            for i in range(eqSteps):  # equilibrate
                mc_move(config, iT)  # Monte Carlo moves

            for i in range(mcSteps):
                mc_move(config, iT)
                val = calcMag(config)  # śrenia wartość spinu
                M1 += abs(val) #magnetyzacja
                S1 += abs(val * val) #podatność
            M[tt] = M1*n1
            S[tt] = N*N*(n1*S1 - n2*M1 * M1) * (1/(iT))

        if magnetization==True:

            if N == 10:
                plt.scatter(t, M, s=30, marker='o')
            elif N == 50:
                plt.scatter(t, M, s=30, marker='v')
            else:
                plt.scatter(t, M, s=10, marker='s')
            plt.plot(t, M)
            plt.xlabel("Reduced Temperature (T*)", fontsize=20)
            plt.ylabel("Magnetization <m>", fontsize=20)
            plt.axis('tight')
            plt.grid()

        else:
            if N == 10:
                plt.scatter(t, S, s=30, marker='o')
            elif N==50:
                plt.scatter(t, S, s=30, marker='v')
            else:
                plt.scatter(t, S, s=10, marker='s')
            plt.plot(t, S)
            plt.xlabel("Reduced Temperature (T*)", fontsize=20)
            plt.ylabel("Susceptibility ", fontsize=20)
            plt.axis('tight')
            plt.grid()

    plt.legend(['L=%d' % L[0], 'L=%d' % L[1], 'L=%d' % L[2]])
    plt.show()

draw(magnetization=False)