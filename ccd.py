from astropy.io import fits
from lsp import lstsq
import numpy as np
from matplotlib import pyplot as plt
import json

##if __name__ == "__main__":
##    pass

def make_plot(x, y, k, b):
    plt.scatter(x, y, label='sigma')
    plt.plot(x, k*x + b, color='r', label='lstsq')
    plt.xlabel('x')
    plt.ylabel('sigma^2')
    plt.legend()
    plt.savefig('ccd.png')

if __name__ == '__main__':
    with fits.open('ccd.fits.gz') as f:
        data = f[0].data.astype(np.int16)
        
        u0 = np.mean(data[0][0])
        x = np.mean(data[:, 0, ...], axis=(1, 2)) - u0
        sigma = np.var((data[:, 0, ...] - data[:, 1, ...]), axis=(1, 2))
        A = np.column_stack((x, np.ones_like(x)))
        x_sol, cost, var = lstsq(A, x, 'ne')
        k, b = x_sol[0], x_sol[1]
        g = 2. / k
        sgm = g * np.sqrt(np.abs(b) / 2)
        g_err = cost * 2 / k**2
        sgm_err = g_err * np.sqrt(np.abs(b) / 2) + g / 2 * np.sqrt(np.abs(b) / 2)
        d = {'ron': sgm, 'ron_err' : sgm_err, 'gain' : g, 'gain_err' : g_err}
        make_plot(x, sigma, k, b)
        with open('ccd.json', 'w') as fp:
            json.dump(d, fp)
