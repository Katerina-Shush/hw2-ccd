from lsp import lstsq
import numpy as np

def lstsq_ne(A, b):
    assert len(A.shape) == len(b.shape) + 1, "Can's solve for shapes {} and {}".format(len(A.shape), len(b.shape))
    assert A.shape[0] == b.shape[0], "Can's solve for shapes {} and {}".format(A.shape, b.shape)
    
    AA = np.linalg.inv(np.dot(A.T, A))
    Ab = np.dot(A.T, b)
    x = np.dot(AA, Ab)
    r = np.dot(A, x) - b
    cost = np.dot(r.T, r)
    sigma = cost / (A.shape[0] - A.shape[1])
    var = A * sigma * sigma
    return x, cost, var

def lstsq_svd(A, b):
    pass

def lstsq(A, b, method, **kwargs):
    if method == 'ne':
        return lstsq_ne(A, b)
    else:
        return lstsq_svd(A, b, **kwargs)

if __name__ == '__main__':
    A = np.random.normal(size=(500, 20))
    params_x = np.random.normal(size=(20,))
    b = A @ params_x + 0.01*np.random.normal(size=(500,))

