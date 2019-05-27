# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:58:02 2019

@author: ross
"""
import numpy as np
import scipy as sp
import LMM as lmm



## Simple soft-thresholding
def softTh(B,Lambda):
    X=np.sign(B)*np.maximum(0,abs(B)-Lambda);
    return X

def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: `numpy array`
            a HSI cube ((m*n) x p)

       noise_type: `string [optional 'additive'|'poisson']`

    Returns: `tuple numpy array, numpy array`
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w=np.zeros((L,N), dtype=np.float)
        RR=np.dot(r,r.T)
        RRi = np.linalg.pinv(RR+small*np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0,i]=0;
            w[i,:] = r[i,:] - np.dot(beta,r)
        Rw = np.diag(np.diag(np.dot(w,w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    #verb = 'poisson'
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w,w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T

def NU_BGBM(Y, E, **kwargs):

    
    # Set the defaults for the optional parameters
    Lambda = 1e-3;
    mu = 1e-2;
    tol = 1e-6;
    maxiter = 1000;
    C, P = Y.shape;
    M = np.size(E, 1);
    # Read the optional parameters
    
    for key in kwargs:
            Ukey = key.upper()
            
            if(Ukey == 'MAXITER'):
                maxiter = np.round(kwargs[key]);
                if (maxiter < 0 ):
                    raise ValueError('maxiter must a positive integer');
            elif(Ukey == 'LAMBDA_P'):
                Lambda = kwargs[key]
                if (np.sum(np.sum(Lambda < 0)) >  0 ):
                    raise ValueError('lambda_p must be positive');
            elif (Ukey =='MU'):
                mu =  kwargs[key]
            elif(Ukey=='TOL'):
                tol = kwargs[key]
            elif(Ukey=='VERBOSE'):
                verbose = kwargs[key]
            else:
                # Hmmm, something wrong with the parameter string
                raise ValueError('Unrecognized option: {}'.format(key))
    
    
    
    ## Initializing optimization variables
    A = lmm.FCLS(Y.T, E.T).T;
    F = Bilinear_endmember( E );
    B = 0.1*Bilinear_abubdance(A);
    S = np.zeros(Y.shape);
    X = Y - S;
    Noise, Rn = est_noise(X, noise_type='additive');
    W = np.diag(1.0/np.sqrt(np.sum(Noise**2, axis=1)/P));
    W1 = W/np.mean(np.diag(W));
    Lambda = Lambda/np.mean(np.mean((W1.dot(Y))**2));
        
    #Initialization of auxiliary matrices V1, V2, V3
    V1 = S.copy();
    V2 = A.copy();
    V3 = B.copy();
    
    #Initialization of Lagranfe Multipliers
    D1 = V1*0;
    D2 = V2*0;
    D3 = V3*0;
    
    #primal residual 
    res_p = float('inf');
    
    #dual residual
    res_d = float('inf');
    
    #indication variable
    mu_changed = 0;
    
    #error tolerance
    tol1 = np.sqrt((3*M + C)*P)*tol;
    
    #current iteration number
    k=1;
    
    ## Main iteration
    while (k < maxiter) and ((abs (res_p) > tol1) or (abs (res_d) > tol1)):
        if (k % 10) == 1:
            #max(res_p,res_d)
            V10 = V1.copy();
            V20 = V2.copy();
            V30 = V3.copy();
        X = Y - S;
        Noise, Rn = est_noise(X, noise_type='additive');
        W = np.diag(1.0/np.sqrt(np.sum(Noise**2, axis=1)/P));
        W1 = W/np.mean(np.diag(W));
        A = np.linalg.inv((W1.dot(E)).T.dot(W1.dot(E)) + mu*np.eye(np.size(E,1))).dot((W1.dot(E)).T.dot(W1).dot(Y - F.dot(B) - V1) + mu*(V2 - D2));
        B = np.linalg.inv((W1.dot(F)).T.dot(W1.dot(F)) + mu*np.eye(np.size(F,1))).dot((W1.dot(F)).T.dot(W1).dot(Y - E.dot(A) - V1) + mu*(V3 - D3));
        S = softTh(V1 - D1, Lambda/mu);
        V1 = np.linalg.inv(W1.T.dot(W1) + mu*np.eye(np.size(W1,1))).dot(W1.T.dot(W1).dot(Y - E.dot(A) - F.dot(B)) + mu*(S + D1));
        V2 = np.maximum(A + D2, 0);
        V3 = np.minimum(np.maximum(B + D3, 0), Bilinear_abubdance( A ));
        
        # Lagrange multipliers update
        D1 = D1 - (V1 - S);
        D2 = D2 - (V2 - A);
        D3 = D3 - (V3 - B);
        
        # update mu so to keep primal and dual residuals whithin a factor of 10
        if (k % 10) == 1:
            #primal residual
            res_p = np.linalg.norm(np.vstack((np.vstack((V1, V2)), V3)) - np.vstack((np.vstack((S, A)), B)), 'fro');
            #dual residual
            res_d = mu*np.linalg.norm(np.vstack((np.vstack((V1, V2)),V3)) - np.vstack((np.vstack((V10, V20)), V30)), 'fro');
            # update mu
            if res_p > 10*res_d:
                mu = mu*2;
                D1 = D1/2;
                D2 = D2/2;
                D3 = D3/2;
                mu_changed = 1;
            elif res_d > 10*res_p:
                mu = mu/2;
                D1 = D1*2;
                D2 = D2*2;
                D3 = D3*2;
                mu_changed = 1;
            
            if  mu_changed:
                mu_changed = 0;
            
        k = k + 1;
    
    if k == maxiter:
        print 'Maximum iteration has been reached!'
     
    A[A<1e-3] = 0;
    B[B<1e-3] = 0;
    Y = E.dot(A) + F.dot(B);
    Results=(A,F,B,S,Y)
    return Results


def Bilinear_endmember( E ):
    M = np.size(E,1);
    num1 = 0;
    F=[]
    for k in range(M-1):
        for j in range(k+1,M):
            F.append(E[:,k]*E[:,j])
            num1=num1+1;
    F=np.asarray(F) .T   
    return F

def Bilinear_abubdance( A ):
    M = np.size(A,0);
    num1 = 0;
    B=[]
    for k in range(M-1):
        for j in range(k+1,M):
            B.append(A[k,:]*A[j,:]);
            num1=num1+1;
    B=np.asarray(B)  
    return B
    
def Metric_RMSE( A_ref, A):
    M, P = A.shape;
    RMSE = np.linalg.norm(A_ref-A,'fro')/(np.sqrt(M.dot(P)));
    return RMSE

#import scipy as sp
#y=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/GBM_unmix_GDA/y.mat")['y']
#MPlus=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/GBM_unmix_GDA/MPlus")['MPlus']
#alpha0=np.array([0.3,0.6,0.1]).T; # Actual abundance vector
#gamma0=np.array([0.5, 0.1, 0.3]).T; # Actual gamma vector
#sigma20= 3*10**-4; # Actual noise variance  
#
#
#lammbda = 10**(-1);
#param = {'LAMBDA_P':lammbda
#             }
#Results = NU_BGBM(y, MPlus,**param);
#print alpha0
#print Results[0]