# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:04:18 2019

@author: ross
"""

import numpy as np


def UCLS(M, U):
    """
    Performs unconstrained least squares abundance estimation.

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).
     """
    Uinv = np.linalg.pinv(U.T)
    return np.dot(Uinv, M[0:,:].T).T


def NNLS(M, U):
    """
    NNLS performs non-negative constrained least squares of each pixel
    in M using the endmember signatures of U.  Non-negative constrained least
    squares with the abundance nonnegative constraint (ANC).
    Utilizes the method of Bro.

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).

    References:
        Bro R., de Jong S., Journal of Chemometrics, 1997, 11, 393-401.
    """
    import scipy.optimize as opt

    N, p1 = M.shape
    q, p2 = U.shape

    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        # opt.nnls() return a tuple, the first element is the result
        X[n1] = opt.nnls(MtM, np.dot(U, M[n1]))[0]
    return X


def _numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.vstack([A1, A2])


def _numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])


def _numpy_to_cvxopt_matrix(A):
    from cvxopt import matrix
    A = np.array(A, dtype=np.float64)
    if A.ndim == 1:
        return matrix(A, (A.shape[0], 1), 'd')
    else:
        return matrix(A, A.shape, 'd')



def hyperFcls( M, U ):
#HYPERFCLS Performs fully constrained least squares on pixels of M.
# hyperFcls performs fully constrained least squares of each pixel in M 
# using the endmember signatures of U.  Fully constrained least squares 
# is least squares with the abundance sum-to-one constraint (ASC) and the 
# abundance nonnegative constraint (ANC).
#
# Usage
#   [ X ] = hyperFcls( M, U )
# Inputs
#   M - HSI data matrix (p x N)
#   U - Matrix of endmembers (p x q)
# Outputs
#   X - Abundance maps (q x N)
#
# References
#   "Fully Constrained Least-Squares Based Linear Unmixing." Daniel Heinz, 
# Chein-I Chang, and Mark L.G. Althouse. IEEE. 1999.

    
    [p1, N] = M.shape;
    [p2, q] = U.shape;
#    if (p1 != p2):
#        print 'M and U must have the same number of spectral bands.';
#        return
#    
    p = p1;
    X = np.zeros((q, N));
    Mbckp = U.copy();
    for n1 in range(N):
        count = q;
        done = 0;
        ref = np.array(range(q));
        r = M[:, [n1]];
        U = Mbckp.copy();
        while not(done):
            als_hat = (np.linalg.inv(U.T.dot(U))).dot(U.T).dot(r);
            s = np.linalg.inv(U.T.dot(U)).dot(np.ones((count, 1)));
    
            # IEEE Magazine method (http://www.planetary.brown.edu/pdfs/3096.pdf)
            # Contains correction to sign.  Error in original paper.
            afcls_hat = als_hat - np.linalg.inv(U.T.dot(U)).dot(np.ones((count, 1))).dot(np.linalg.inv(np.ones((1, count)).dot(np.linalg.inv(U.T.dot(U))).dot(np.ones((count, 1))))).dot(np.ones((1, count)).dot(als_hat)-1);
    
            # See if all components are positive.  If so, then stop.
            if (np.sum(afcls_hat>0) == count):
                alpha = np.zeros((q, 1));
                alpha[ref] = afcls_hat;
                break;
            
            # Multiply negative elements by their counterpart in the s vector.
            # Find largest abs(a_ij, s_ij) and remove entry from alpha.
            idx = np.nonzero(afcls_hat<0)[0];
            afcls_hat[idx] = afcls_hat[idx] / s[idx];
            maxIdx = np.argmax(abs(afcls_hat[idx]));
            if np.isscalar(idx):
                maxIdx=idx
            else:
                maxIdx = idx[maxIdx];
#            alpha = np.zeros((q, 1));
#            alpha[maxIdx] = 0;
            keep = np.setdiff1d(range(np.size(U, 1)), maxIdx);
            U = U[:, keep];
            count = count - 1;
            ref = ref[keep];
        X[:, [n1]] = alpha;
    return X
    
    
def FCLS(M, U):
    """
    Performs fully constrained least squares of each pixel in M
    using the endmember signatures of U. Fully constrained least squares
    is least squares with the abundance sum-to-one constraint (ASC) and the
    abundance nonnegative constraint (ANC).

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).

    References:
         Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
         Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.

    Notes:
        Three sources have been useful to build the algorithm:
            * The function hyperFclsMatlab, part of the Matlab Hyperspectral
              Toolbox of Isaac Gerg.
            * The Matlab (tm) help on lsqlin.
            * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
              http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
              , it's great code.
    """
    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False
    N, p1 = M.shape
    nvars, p2 = U.shape

    C = _numpy_to_cvxopt_matrix(U.T)
    Q = C.T * C

    lb_A = -np.eye(nvars)
    lb = np.repeat(0, nvars)
    A = _numpy_None_vstack(None, lb_A)
    b = _numpy_None_concatenate(None, -lb)
    A = _numpy_to_cvxopt_matrix(A)
    b = _numpy_to_cvxopt_matrix(b)

    Aeq = _numpy_to_cvxopt_matrix(np.ones((1,nvars)))
    beq = _numpy_to_cvxopt_matrix(np.ones(1))

    M = np.array(M, dtype=np.float64)
    X = np.zeros((N, nvars), dtype=np.float32)
    for n1 in range(N):
        d = matrix(M[n1], (p1, 1), 'd')
        q = - d.T * C
        sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)['x']
        X[n1] = np.array(sol).squeeze()
    return X