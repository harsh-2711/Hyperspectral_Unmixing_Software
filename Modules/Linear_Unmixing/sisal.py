# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:43:14 2019

@author: ross
"""
# General libraries
import sys
import warnings
# Numpy
import numpy as np
import numpy.linalg as lin
# Scipy
import scipy as sp
import scipy.linalg as splin
import scipy.sparse.linalg as slin
# Matplotlib
import matplotlib.pyplot as plt
#############################################
# SISAL algorithm
#############################################
from vca import *


def soft_neg(y,tau) :
    """  z = soft_neg(y,tau);
    
      negative soft (proximal operator of the hinge function)
    """

    z = np.maximum(np.abs(y+tau/2) - tau/2, 0)
    z = z*(y+tau/2)/(z+tau/2)
    return z

def sisal(Y,p,**kwargs):
    """
    M,Up,my,sing_values = sisal(Y,p,**kwargs)
        
    ----- Description ---------------
        
    Simplex identification via split augmented Lagrangian (SISAL) estimates 
    the vertices  M={m_1,...m_p} of the (p-1)-dimensional simplex of minimum 
    volume containing the vectors [y_1,...y_N], under the assumption that y_i
    belongs to a (p-1)  dimensional affine set.
    
    For details see
    
    [1] José M. Bioucas-Dias, "A variable splitting augmented lagrangian
    approach to linear spectral unmixing", First IEEE GRSS Workshop on 
    Hyperspectral Image and Signal Processing - WHISPERS, 2009. 
    http://arxiv.org/abs/0904.4635v1
    
    
    ----- Input ---------------------
    
    Y - matrix with dimension  L(channels) x N(pixels). Each pixel is a linear
        mixture of p endmembers signatures Y = M*x + noise.
    
    p - number of independent columns of M. Therefore, M spans a (p-1)-dimensional
        affine set. p is the number of endmembers.
    
    ----- Optional input ------------
    
    
    mm_iters - Maximum number of constrained quadratic programs
               Default: 80
    
    tau - Regularization parameter in the problem
             Q^* = arg min_Q  -\log abs(det(Q)) + tau*|| Q*yp ||_h
                   subject to np.ones((1,p))*Q=mq
             where mq = ones(1,N)*yp'inv(yp*yp) and ||x||_h is the "hinge"
             induced norm (see [1]).
          Default: 1
    
    mu - Augmented Lagrange regularization parameter
         Default: 1
    
    spherize - {True, False} Applies a spherization step to data such that the spherized
               data spans over the same range along any axis.
               Default: True
    
    tolf - Tolerance for the termination test (relative variation of f(Q))
           Default: 1e-2
    
    M0 - Initial M, dimension L x p. 
         Defaults is given by the VCA algorithm.
    
    verbose - {0,1,2,3} 
                    0 - work silently
                    1 - display simplex volume
                    2 - display figures
                    3 - display SISAL information 
                    4 - display SISAL information and figures
              Default: 1
    
    ----- Output --------------------
    
    M - estimated endmember signature matrix L x p
    
    Up - isometric matrix spanning the same subspace as M, imension is L x p
    
    my - mean value of Y
    
    sing_values - (p-1) eigenvalues of Cy = (y-my)*(y-my)/N. The dynamic range
                  of these eigenvalues gives an idea of the  difficulty of the
                  underlying problem
    
    ----- Note ----------------------
    
    The identified affine set is given by
           {z\in R^p : z=Up(:,1:p-1)*a+my, a\in R^(p-1)}
    
    
    ----- License -------------------
    Author: Etienne Monier (etienne.monier@enseeiht.fr)
    
    This code is a translation of a matlab code provided by 
    Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
    available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
    Translation of last version at 20-April-2018 (Matlab version 2.1 (7-May-2004))
       
    """
    
    #
    # -------------------------------------------------------------------------
    #
    #
    #--------------------------------------------------------------
    # test for number of required parametres
    #--------------------------------------------------------------
    
    # data set size
    L,N = Y.shape
    if (L<p) :
        raise ValueError('Insufficient number of columns in y')
    
    ##
    #--------------------------------------------------------------
    # Set the defaults for the optional parameters
    #--------------------------------------------------------------
    # maximum number of quadratic QPs
    
    MMiters = 80
    spherize = True
    # display only volume evolution
    verbose = 1
    # soft constraint regularization parameter
    tau = 1.0
    # Augmented Lagrangian regularization parameter
    mu = p*1000.0/N
    # no initial simplex
    M = 0.0
    # tolerance for the termination test
    tol_f = 1e-2
    
    ##
    #--------------------------------------------------------------
    # Local variables
    #--------------------------------------------------------------
    # maximum violation of inequalities
    slack = 1e-3
    # flag energy decreasing
    energy_decreasing = 0.0
    # used in the termination test
    f_val_back = float("inf")
    #
    # spherization regularization parameter
    lam_sphe = 1e-8
    # quadractic regularization parameter for the Hesssian
    # Hreg = = mu*I
    lam_quad = 1e-6
    # minimum number of AL iterations per quadratic problem 
    AL_iters = 4
    # flag 
    flaged = 0
    
    #--------------------------------------------------------------
    # Read the optional parameters
    #--------------------------------------------------------------
    
    for key in kwargs:
        Ukey = key.upper()
        
        if(Ukey == 'MM_ITERS'):
            MMiters = kwargs[key]
        elif(Ukey == 'SPHERIZE'):
            spherize = kwargs[key]
        elif (Ukey =='MU'):
            mu =  kwargs[key]
        elif (Ukey=='TAU'):
            tau = kwargs[key]
        elif(Ukey=='TOLF'):
            tol_f = kwargs[key]
        elif(Ukey=='M0'):
            M = kwargs[key]
        elif(Ukey=='VERBOSE'):
            verbose = kwargs[key]
        else:
            # Hmmm, something wrong with the parameter string
            raise ValueError('Unrecognized option: {}'.format(key))
    
    
    ##
    #--------------------------------------------------------------
    # set display mode
    #--------------------------------------------------------------
    if (verbose == 3) or (verbose == 4):
        warnings.filterwarnings("ignore")
    else :
        warnings.filterwarnings("always")
    
    
    ##
    #--------------------------------------------------------------
    # identify the affine space that best represent the data set y
    #--------------------------------------------------------------
    my = np.mean(Y,axis=1)
    My = np.repeat(my[:,np.newaxis],N,axis=1)
    Myp = np.repeat(my[:,np.newaxis],p,axis=1)
    
    Y = Y-My
    
    Up,d = slin.svds(Y.dot(Y.T)/N,k=p-1,which='LM',v0=np.ones((Y.shape[0]))); # step 3eigenValues, eigenVectors
    d=d[::-1]
    Up=Up[:,::-1]
#    Up,d,_ = lin.svd(Y.dot(Y.T)/N)
#    sort_ind = np.argsort(d)[::-1]
#    Up = Up[:,sort_ind[:p-1]]
#    d = d[sort_ind[:p-1]]
#    
    # represent y in the subspace R^(p-1)
    Y = Up.dot(Up.T).dot(Y)
    # lift y
    Y = Y + My
    # compute the orthogonal component of my
    my_ortho = my-Up.dot(Up.T).dot(my)
    # define another orthonormal direction
    Up = np.concatenate((Up, (my_ortho/np.sqrt(np.sum(my_ortho**2)))[:,np.newaxis] ),axis=1)
    sing_values = d
    
    # get coordinates in R^p
    Y = Up.T.dot(Y)
    
    
    ##
    #------------------------------------------
    # spherize if requested
    #------------------------------------------
    if spherize:
        Y = Up.dot(Y)
        Y = Y-My
        C = np.diag(1/np.sqrt(d+lam_sphe))
        IC = lin.inv(C)
        Y=C.dot(np.transpose(Up[:,:p-1])).dot(Y)
        # lift
        Y = np.concatenate((Y,np.ones((1,N))),axis=0)
        #Y[p-1,:] = 1
        # normalize to unit norm
        Y = Y/np.sqrt(p)
    
    
    ##
    # ---------------------------------------------
    #            Initialization
    #---------------------------------------------
    if M == 0:
        # Initialize with VCA
        Mvca,_,_ = vca(Y,p,verbose=False)
        M = Mvca.copy()
        # expand Q
        Ym = np.mean(M,axis=1)
        Ym = np.repeat(Ym[:,np.newaxis],p,axis=1)
        dQ = M - Ym 
        # fraction: multiply by p is to make sure Q0 starts with a feasible
        # initial value.
        M = M + p*dQ
    else:
        # Ensure that M is in the affine set defined by the data
        M = M-Myp
        M = Up[:,:p-1].dot(Up[:,:p-1].T).dot(M)
        M = M + Myp
        M = (Up.T).dot(M)   # represent in the data subspace
        # is sherization is set
        if spherize:
            M = Up.dot(M)-Myp
            M=C.dot(Up[:,:p-1].T).dot(M)
            # lift
            M[p-1,:] = 1
            # normalize to unit norm
            M = M/np.sqrt(p)
    
    
    Q0 = lin.inv(M)
    Q=Q0.copy()
    
    
    # plot  initial matrix M
    if verbose == 2 or verbose == 4 :
        
        M = lin.inv(Q)
        fig,ax = plt.subplots()
        
        line1 = ax.plot(Y[0,:],Y[1,:],'.')
        line2 = ax.plot(M[0,:], M[1,:],'ok')
    
        ax.set_title('SISAL: Endmember Evolution')
        
    
    #
    # ---------------------------------------------
    #            Build constant matrices
    #---------------------------------------------
    
    AAT = np.kron(Y.dot(Y.T),np.eye(p))     # size p^2xp^2
    B = np.kron(np.eye(p),np.ones((1,p)))   # size pxp^2
    qm = np.sum(lin.inv((Y.dot(Y.T))).dot(Y),axis=1)
    
    H = lam_quad*np.eye(p**2)
    F = H+mu*AAT              # equation (11) of [1]
    IF = lin.inv(F)
    
    # auxiliar constant matrices
    G = IF.dot(B.T).dot(lin.inv(B.dot(IF).dot(B.T)))
    qm_aux = G.dot(qm)
    G = IF-G.dot(B).dot(IF)
    
   
    ##
    # ---------------------------------------------------------------
    #          Main body- sequence of quadratic-hinge subproblems
    #----------------------------------------------------------------
    
    # initializations
    Z = Q.dot(Y)
    Bk = 0*Z
    
#    Z=sp.io.loadmat("/home/ross/Codes/ReferenceCodes/HSMSFusionToolbox/Methods/Lanaras/sisal/Z.mat")['Z']
#    Q=sp.io.loadmat("/home/ross/Codes/ReferenceCodes/HSMSFusionToolbox/Methods/Lanaras/sisal/Q.mat")['Q']
#    Y=sp.io.loadmat("/home/ross/Codes/ReferenceCodes/HSMSFusionToolbox/Methods/Lanaras/sisal/Y.mat")['Y']
#    
    hinge = lambda x: np.maximum(-x,0)
    
    
    for k in range(MMiters):
        
        IQ = lin.inv(Q)
        g = -IQ.T
        g = g.flatten(order='C')
    
        baux = H.dot(Q.flatten(order='C'))-g
    
        q0 = Q.flatten(order='C')
        Q0 = Q.copy()
        
        # display the simplex volume
        if verbose == 1 :
            if spherize:
                # unscale
                M = IQ*np.sqrt(p)
                # remove offset
                M = M[:p-1,:]
                # unspherize
                M = Up[:,:p-1].dot(IC).dot(M)
                # sum ym
                M = M + Myp
                M = Up.T.dot(M)
            else:
                M = IQ
            
            print('\n iter = {0}, simplex volume = {1:.4e}  \n'.format(k, 1/np.abs(lin.det(M))))
    
        
        if k == MMiters-1:
            AL_iters = 100
                
        
        while 1 :
            q = Q.flatten(order='C')
            # initial function values (true and quadratic)
            f0_val = -np.log(np.abs(lin.det(Q)))+ tau*np.sum(hinge(Q.dot(Y)))
            f0_quad = (q-q0).T.dot(g)+0.5*(q-q0).T.dot(H).dot(q-q0) + tau*np.sum(hinge(Q.dot(Y)))
#            i=2:AL_iters
            for i in range(1,AL_iters):
                #-------------------------------------------
                # solve quadratic problem with constraints
                #-------------------------------------------
                dq_aux= Z+Bk                # matrix form
                dtz_b = dq_aux.dot(Y.T)
                dtz_b = dtz_b.flatten(order='C')
                b = baux+mu*dtz_b           # (11) of [1]
                q = G.dot(b)+qm_aux         # (10) of [1]
                Q = np.reshape(q,(p,p),order='C')
                
                #-------------------------------------------
                # solve hinge
                #-------------------------------------------
                Z = soft_neg(Q.dot(Y)-Bk,tau/mu);
                
                #-------------------------------------------
                # update Bk
                #-------------------------------------------
                
                Bk = Bk - (Q.dot(Y)-Z)
                if verbose == 3 or  verbose == 4 :
                    print('\n ||Q*Y-Z|| = {0:.4f}'.format(lin.norm(Q.dot(Y)-Z,'fro')))
                
                if verbose == 2 or verbose == 4:
                    M = lin.inv(Q)
                    line2.set_xdata(M[0,:])
                    line2.set_ydata(M[1,:])
                    plt.draw()
                    if ~flaged :
                         line3 = ax.plot(M[0,:], M[1,:],'.r')
                         legend( 'data points','M(0)','M(k)')
                         flaged = 1
            
            
            f_quad = (q-q0).T.dot(g)+0.5*(q-q0).T.dot(H).dot(q-q0) + tau*np.sum(hinge(Q.dot(Y)))
            if verbose == 3 or  verbose == 4:
                print('\n MMiter = {0}, AL_iter, = {1},  f0 = {2:2.4f}, f_quad = {3:2.4f},  \n'.format(k,i, f0_quad,f_quad))
    
            f_val = -np.log(np.abs(lin.det(Q)))+ tau*np.sum(hinge(Q.dot(Y)))
            if f0_quad >= f_quad:            # quadratic energy decreased
                try:
                    while  f0_val < f_val :
                        if verbose == 3 or  verbose == 4 :
                            print('\n line search, MMiter = {0}, AL_iter, = {1},  f0 = {2:2.4f}, f_val = {3:2.4f},  \n'.format(k,i, f0_val,f_val))
    
                    # do line search
                        Q = (Q+Q0)/2
                        f_val = -np.log(np.abs(lin.det(Q)))+ tau*sum(hinge(Q.dot(Y)))
                    break
                except:
                    1+1
    
    
    if verbose == 2 or verbose == 4:
        
        ax.legend('data points','M(0)',  'M(final)')
        
#        p_H(4) = plot(M(1,:), M(2,:),'*g');
#        leg_cell{end+1} = ;
#        legend(p_H', leg_cell);
#    end
    
    
    if spherize :
        M = lin.inv(Q)
        # refer to the initial affine set
        # unscale
        M = M*np.sqrt(p)
        # remove offset
        M = M[:p-1,:]
        # unspherize
        M = Up[:,:p-1].dot(IC).dot(M)
        # sum ym
        M = M + Myp
    else :
        M = Up.dot(lin.inv(Q))
    
    
    return (M,Up,my,sing_values)
