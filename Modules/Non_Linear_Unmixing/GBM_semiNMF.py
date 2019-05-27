# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:59:44 2019

@author: ross
"""
import numpy as np
import scipy as sp
import LMM as lmm


def UPDATE_W(data,W,H):
    W1 = np.dot(data[:,:], H.T)
    W2 = np.dot(H, H.T)    
    W = np.dot(W1, np.linalg.inv(W2))
    return W
        
def UPDATE_H(data,W,H):
    def separate_positive(m):
        return (np.abs(m) + m)/2.0 
    
    def separate_negative(m):
        return (np.abs(m) - m)/2.0
        
    XW = np.dot(data[:,:].T, W)                

    WW = np.dot(W.T, W)
    WW_pos = separate_positive(WW)
    WW_neg = separate_negative(WW)
    
    XW_pos = separate_positive(XW)
    H1 = (XW_pos + np.dot(H.T, WW_neg)).T
    
    XW_neg = separate_negative(XW)
    H2 = (XW_neg + np.dot(H.T, WW_pos)).T + 10**-9
    
    H *= np.sqrt(H1/H2)  
    return H


def factorize(data,W,H,niter=100,tol_inner=1e-6):
#        EPS = np.finfo(float).eps
        ferr = np.zeros(niter)
             
        for i in xrange(niter):
            W=UPDATE_W(data.copy(),W.copy(),H.copy())
            H=UPDATE_H(data.copy(),W.copy(),H.copy())         
            ferr[i] = np.sqrt( np.sum((data[:,:] - np.dot(W, H))**2 ))   ###  frobenius_norm        
#            print 'Error : (%s %s)'  %(ferr[i], i)  
            # check if the err is not changing anymore
            derr = np.abs(ferr[i] - ferr[i-1])/ferr[i]
            if i > 1 and derr<tol_inner:
                ferr = ferr[:i]
                break
        return H

def separate_positive(m):
    return (np.abs(m) + m)/2.0 
    
def separate_negative(m):
    return (np.abs(m) - m)/2.0

def GBM_semiNMF(Y, E, tol=1e-6, maxiter_iter=1000,verbose='on'):

    ## Initializing optimization variables
    A = lmm.FCLS(Y.T, E.T).T;
    M = Bilinear_endmember( E );
#    B = 0.1*Bilinear_abubdance(A);
    k = 1;
    [L, P] = A.shape;
    cost0 = 0
    ## Main iteration
    while k < maxiter_iter:
        A0 = A.copy();
        B = 0.01*Bilinear_abubdance(A);
        B0=B.copy()
        
        #Update A starts here
        Y1=E.dot(A0)
        Y1E = np.dot(Y1[:,:].T, E) 
        EE = np.dot(E.T, E)
        Y1E_pos=separate_positive(Y1E)
        Y1E_neg=separate_negative(Y1E)
        
        EE_pos=separate_positive(EE)
        EE_neg=separate_negative(EE)
        A1=(Y1E_pos + np.dot(A0.T, EE_neg)).T
        A2=(Y1E_neg + np.dot(A0.T, EE_pos)).T #+10**-9
        A*=np.sqrt(A1/A2)
        

        #Update A starts here
        Y2=M.dot(B0)
        Y2M = np.dot(Y2[:,:].T, M) 
        MM = np.dot(M.T, M)
        Y2M_pos=separate_positive(Y2M)
        Y2M_neg=separate_negative(Y2M)
        
        MM_pos=separate_positive(MM)
        MM_neg=separate_negative(MM)
        B1=(Y2M_pos + np.dot(B0.T, MM_neg)).T
        B2=(Y2M_neg + np.dot(B0.T, MM_pos)).T #+10**-9
        B*=np.sqrt(B1/B2)
        mask=B>B0
        B[mask]=B0[mask]
        
        cost = np.sum((Y- (np.dot(E,A)+ np.dot(M,B)))**2)
        if k > 2 and (cost0-cost)/cost < tol:
            if verbose == 'on':
                print 'Initialization of H_hyper converged at the ', k, 'th iteration '
            break
        cost0 = cost
        
#        ratio = abs(Obj_func(Y, E, A, M, B) - Obj_func(Y, E, A0, M, B))/abs(Obj_func(Y, E, A, M, B));
#        print 'Ratio : (%s %s)'  %(ratio, k)  
#        if ratio < tol_outer:
#            break;
        k = k + 1;
    
    if k == maxiter_iter:
        print 'Maximum iteration has been reached!'
    
    RMSE_h = (cost0/(y.shape[1]*y.shape[0]))**0.5
    if verbose == 'on':
        print ' RMSE = ', RMSE_h
        
        
    Y = E.dot(A) + M.dot(B);
    Results=(A,M,B,Y);
    
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


def Obj_func(Y, E, A, F, B):
    V = np.linalg.norm(Y - E.dot(A) - F.dot(B), 'fro')**2/2;
    return V

def Metric_RMSE( A_ref, A):
    M, P = A.shape;
    RMSE = np.linalg.norm(A_ref-A,'fro')/(np.sqrt(M*P));
    return RMSE

#import scipy as sp
#y=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/GBM_unmix_GDA/y.mat")['y']
#MPlus=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/GBM_unmix_GDA/MPlus")['MPlus']
#alpha0=np.array([0.3,0.6,0.1]).T; # Actual abundance vector
#gamma0=np.array([0.5, 0.1, 0.3]).T; # Actual gamma vector
#sigma20= 3*10**-4; # Actual noise variance  
#
##
##Y=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/Y.mat")['Y']
##E=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/E.mat")['E']
##A=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/A.mat")['A']
##F=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/F.mat")['F']
##B=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/B.mat")['B']
#
#Results = GBM_semiNMF(y, MPlus, 1e-6, 10);
#print alpha0
#print Results[0]
#RMSE = Metric_RMSE(A, Results[0]);
#print RMSE


