# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:59:44 2019

@author: ross
"""
import numpy as np
import scipy as sp
import LMM as lmm



## Projected Gradient
def Projected_Gradient(gradH, H, LB, UB):
    row, col = H.shape;
    Proj_grad=np.zeros(H.shape)
    for j in range(row):
        for k  in range(col):
            if (H[j,k] > LB[j,k]) and (H[j,k] < UB[j,k]):
                Proj_grad[j,k] = gradH[j,k];
            if  H[j,k] == LB[j,k]:
                Proj_grad[j,k] = np.min(gradH[j,k],0);
            
            if  H[j,k] == UB[j,k]:
                Proj_grad[j,k] = np.max(gradH[j,k],0);
            
    Proj_grad_norm = np.linalg.norm(Proj_grad, 'fro');
    
    return Proj_grad_norm


def OGM_NLS_Bound(Z, W, H, LB, UB, tol, maxiter):
    
    ## Initializing variables
    alpha = 1;
    L = np.linalg.norm(W.T.dot(W),ord=2);
    V = H.copy();
    k = 1;
    gradH = W.T.dot(W).dot(H) - W.T.dot(Z);
    init_delta = Projected_Gradient(gradH, H, LB, UB);
    delta = float('inf');
    
    ## Main iteration
    while (k < maxiter) and (delta > max(10**(-3), tol)*init_delta):
        gradH = W.T.dot(W).dot(H) - W.T.dot(Z);
        delta = Projected_Gradient(gradH, H, LB, UB);
        H0 = H.copy();
        H = np.minimum(np.maximum((V - gradH/L), LB), UB);
        alpha0 = alpha;
        alpha = (1 + np.sqrt(4*alpha0**2+1))/2;
        V = H + ((alpha0 - 1)/alpha)*(H - H0);
        k = k + 1;
    
    return H

def GBM_BPOGM(Y, E, tol_inner, maxiter_inner, tol_outer, maxiter_outer):

    ## Initializing optimization variables
    A = lmm.FCLS(Y.T, E.T).T;
    F = Bilinear_endmember( E );
    B = np.zeros((np.size(F,1), np.size(Y, 1)));
    delta = 1;
    k = 1;
    [M, P] = A.shape;
    
    ## Main iteration
    while k < maxiter_outer:
        A0 = A.copy();
        Y1_aug = np.vstack((Y - F.dot(B), delta*np.ones((1,P))));
        E_aug = np.vstack((E, delta*np.ones((1,M))));
        A = OGM_NLS_Bound(Y1_aug, E_aug, A0, np.zeros(A0.shape), float('inf')*np.ones(A0.shape), tol_inner, maxiter_inner);
        B0 = B.copy();
        B = OGM_NLS_Bound(Y - E.dot(A), F, B0, np.zeros(B0.shape), Bilinear_abubdance( A ), tol_inner, maxiter_inner);
        ratio = abs(Obj_func(Y, E, A, F, B) - Obj_func(Y, E, A0, F, B0))/abs(Obj_func(Y, E, A, F, B));
#        print 'Ratio : (%s %s)'  %(ratio, k)  
        if ratio < tol_outer:
            break;
        k = k + 1;
    
    if k == maxiter_outer:
        print 'Maximum iteration has been reached!'
    
    Y = E.dot(A) + F.dot(B);
    Results=(A,F,B,Y);
    
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
#
##Y=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/Y.mat")['Y']
##E=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/E.mat")['E']
##A=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/A.mat")['A']
##F=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/F.mat")['F']
##B=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/BPOGM-master/GBM based Unmixing of Hyperspectral Data Using Bound Projected Optimal Gradient Method/B.mat")['B']
#Results = GBM_BPOGM(y, MPlus, 1e-6, 10, 1e-6, 1000);
#print alpha0
#print Results[0]

#RMSE = Metric_RMSE(A, Results[0]);
#print RMSE