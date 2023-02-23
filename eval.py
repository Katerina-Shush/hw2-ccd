from lsp import lstsq
import numpy as np

if __name__ == '__main__':
    A = np.random.normal(size=(500, 20))
    params_x = np.random.normal(size=(20,))
    b = A @ params_x + 0.01*np.random.normal(size=(500,))
    
