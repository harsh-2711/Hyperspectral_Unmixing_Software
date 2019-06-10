# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:43:06 2019

@author: ross
"""
# General libraries
import sys
import warnings
# Numpy
import numpy as np
import numpy.linalg as lin
import math

# Scipy
import scipy as sp
import scipy.linalg as splin
import scipy.sparse.linalg as slin
from Modules.End_Member_Extraction import eea
# Matplotlib
import matplotlib.pyplot as plt

#############################################
# SUNSAL algorithm
#############################################

class SUNSALModule():
	
	def SUNSAL(self, M, y, **kwargs):
	    """
	    x = sunsal_v2(M,Y,**kwargs)
	        
	    ----- Description ---------------
	        
	    SUNSAL (sparse unmixing via variable splitting and augmented Lagrangian 
	    methods) algorithm implementation. Accepted constraints are:
	        1. Positivity:  X >= 0
	        2. Addone:      np.sum(X,axis=0) = np.ones(N)
	    
	    For details see
	    
	    [1] J. Bioucas-Dias and M. Figueiredo, “Alternating direction algorithms
	    for constrained sparse regression: Application to hyperspectral unmixing”,
	    in 2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal
	    Processing-WHISPERS'2010, Raykjavik, Iceland, 2010.
	    
	    
	    ----- Input ---------------------
	    
	    M - endmember signature matrix with dimensions L(channels) x p(endmembers)
	    
	    Y - matrix with dimensions L(channels) x N(pixels). Each pixel
	        is a linear mixture of p endmembers signatures
	    
	    ----- Optional input ------------
	    
	    al_iters - Minimum number of augmented Lagrangian iterations
	               Default: 100
	    
	    lambda_p - regularization parameter. lambda is either a scalar
	               or a vector with N components (one per column of x)
	               Default: 0
	    
	    
	    positivity - {True, False} Enforces the positivity constraint
	                 Default: False
	    
	    addone - {True, False} Enforces the addone constraint
	             Default: False
	    
	    tol - tolerance for the primal and  dual residuals
	          Default: 1e-4
	    
	    
	    verbose = {True, False}
	              False - work silently
	              True - display iteration info
	              Default: True
	    
	    ----- Output --------------------
	    
	    X - estimated abundance matrix of size p x N
	    
	    ----- License -------------------
	    Author: Etienne Monier (etienne.monier@enseeiht.fr)
	    
	    This code is a translation of a matlab code provided by 
	    Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
	    available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
	    Translation of last version at 20-April-2018 (Matlab version 2.1 (7-May-2004))
	       
	    """
	    
	    
	    #--------------------------------------------------------------
	    # test for number of required parametres
	    #--------------------------------------------------------------
	    
	    # mixing matrixsize
	    LM,p = M.shape
	    # data set size
	    L,N = y.shape
	    if (LM != L):
	        raise ValueError('mixing matrix M and data set y are inconsistent')
	    
	    
	    #--------------------------------------------------------------
	    # Set the defaults for the optional parameters
	    #--------------------------------------------------------------
	    
	    # maximum number of AL iteration
	    AL_iters = 1000
	    # regularizatio parameter
	    Lambda = 0.0
	    # display only sunsal warnings
	    verbose = True
	    # Positivity constraint
	    positivity = False
	    # Sum-to-one constraint
	    addone = False
	    # tolerance for the primal and dual residues
	    tol = 1e-4;
	    # initialization
	    x0 = 0;
	    
	    
	    #--------------------------------------------------------------
	    # Local variables
	    #--------------------------------------------------------------
	    #--------------------------------------------------------------
	    # Read the optional parameters
	    #--------------------------------------------------------------
	    
	    
	    for key in kwargs:
	        Ukey = key.upper()
	        
	        if(Ukey == 'AL_ITERS'):
	            AL_iters = np.round(kwargs[key]);
	            if (AL_iters < 0 ):
	                raise ValueError('AL_iters must a positive integer');
	        elif(Ukey == 'LAMBDA_P'):
	            Lambda = kwargs[key]
	            if (np.sum(np.sum(Lambda < 0)) >  0 ):
	                raise ValueError('lambda_p must be positive');
	        elif (Ukey =='POSITIVITY'):
	            positivity =  kwargs[key]
	        elif (Ukey=='ADDONE'):
	            addone = kwargs[key]
	        elif(Ukey=='TOL'):
	            tol = kwargs[key]
	        elif(Ukey=='VERBOSE'):
	            verbose = kwargs[key]
	        elif(Ukey=='X0'):
	            x0 = kwargs[key]
	            if (x0.shape[0] != p) | (x0.shape[1] != N):
	                raise ValueError('initial X is  inconsistent with M or Y');
	        elif(Ukey=='X_SOL'):
	            X_sol = kwargs[key]
	        elif(Ukey=='CONV_THRE'):
	            conv_thre = kwargs[key]
	        else:
	            # Hmmm, something wrong with the parameter string
	            raise ValueError('Unrecognized option: {}'.format(key))


	    #---------------------------------------------
	    #  If lambda is scalar convert it into vector
	    #---------------------------------------------
	    Nlambda = np.array(Lambda).size
	    if Nlambda == 1:
	#    % same lambda for all pixels
	        Lambda = Lambda*np.ones((p,N));
	    elif Nlambda != N:
	        raise ValueError('Lambda size is inconsistent with the size of the data set');
	    else:
	        # each pixel has its own lambda
	        Lambda = np.repeat(Lambda[np.newaxis,:],p,axis=0)
	#
	#% compute mean norm
	    norm_y = np.sqrt(np.mean(np.mean(y**2)));
	#% rescale M and Y and lambda
	    M = M/norm_y;
	    y = y/norm_y;
	    Lambda = Lambda/norm_y**2;
	    
	    
	    ##
	    #---------------------------------------------
	    # just least squares
	    #---------------------------------------------
	    if (np.sum(Lambda == 0) and (positivity=='no') and (addone=='no')):
	        z = lin.pinv(M).dot(y)
	        # primal and dual residues
	        res_p = 0
	        res_d = 0
	        i = 0
	        return (z,res_p,res_d,i)
	    
	    #---------------------------------------------
	    # least squares constrained (sum(x) = 1)
	    #---------------------------------------------
	    SMALL = 1e-12;
	    B = np.ones((1,p));
	    a = np.ones((1,N));

	    if ((addone=='yes') and (positivity=='no')):
	        F = M.T.dot(M);
	        # test if F is invertible
	        if np.rcond(F) > SMALL:
	            # compute the solution explicitly
	            IF = np.linalg.inv(F);
	            z = IF.dot(M).T.dot(y)-IF.dot(B).T.dot((np.linalg.inv(B.dot(IF).dot(B.T)).dot(B.dot(IF).dot(M).T.dot(y)-a)));
	            # primal and dual residues
	            res_p = 0;
	            res_d = 0;
	            return (z,res_p,res_d,i)
	    
	    
	    ##
	    #---------------------------------------------
	    #  Constants and initializations
	    #---------------------------------------------
	    mu_AL = 0.01;
	    mu = 10*np.mean(Lambda.flatten()) + mu_AL;
	#
	    #%F = M'*M+mu*eye(p);
	    UF,sF,VF= np.linalg.svd(np.dot(M.T,M));
	#    sF = SF.copy();
	    IF = UF.dot(np.diag(1.0/(sF+mu))).dot((UF).T);
	    #%IF = inv(F);
	    Aux = IF.dot((B.T)).dot(np.linalg.inv(B.dot(IF).dot((B.T))));
	    x_aux = Aux.dot(a);
	    IF1 = (IF-Aux.dot(B).dot(IF));
	    
	    yy = np.dot(M.T,y);
	    
	#% no intial solution supplied
	    if x0 == 0:
	        x= np.dot(IF,np.dot(M.T,y));
	        
	    z = x.copy();
	#% scaled Lagrange Multipliers
	    d  = 0*z;
	    
	    #
	    #---------------------------------------------
	    #  AL iterations - main body
	    #---------------------------------------------
	    
	    tol1 = np.sqrt(np.dot(N,p))*tol;
	    tol2 = np.sqrt(np.dot(N,p))*tol;
	    i=1;
	    res_p = float('inf')#math.inf;
	    res_d = float('inf')#math.inf;
	    maskz = np.ones(z.shape);
	    mu_changed = 0;
	    while ((i <= AL_iters) and ((abs (res_p) > tol1) or (abs (res_d) > tol2))): 
	#    % save z to be used later
	        if (i%10) == 1:
	            z0 = z.copy();
	#    % minimize with respect to z
	        z = self.soft(x-d,Lambda/mu);
	#    % teste for positivity
	        if positivity=='yes':
	            maskz = (z >= 0);
	            z = z*maskz; 
	        
	#    % teste for sum-to-one 
	        if addone=='yes':
	           x = IF1.dot((yy+mu*(z+d)))+x_aux;
	        else:
	           x = IF.dot((yy+mu*(z+d)));
	        
	#    % Lagrange multipliers update
	        d = d-(x-z);
	#
	#    % update mu so to keep primal and dual residuals whithin a factor of 10
	        if (i%10) == 1:
	#        % primal residue
	            res_p = np.linalg.norm(x-z,'fro');
	#        % dual residue
	            res_d = mu*np.linalg.norm(z-z0,'fro');
	            #print('in if modddddd',res_p,res_d)
	            if  verbose=='yes':
	                print(f'i = {i}, res_p = {res_p} res_d = {res_d}')
	#        % update mu
	            if res_p > 10*res_d:
	                mu = mu*2;
	                d = d/2;
	                mu_changed = 1;
	            elif res_d > 10*res_p:
	                mu = mu/2;
	                d = d*2;
	                mu_changed = 1;
	            if  mu_changed:
	           #% update IF and IF1
	               IF = np.dot(UF,np.dot(np.diag(1/(sF+mu)),UF.T));
	               Aux = np.dot(IF,np.dot(B.T,np.linalg.inv(np.dot(B,np.dot(IF,B.T)))));
	               x_aux = np.dot(Aux,a);
	               IF1 = (IF-np.dot(Aux,np.dot(B,IF)));
	               mu_changed = 0;
	        i=i+1;
	            
	    #########################################################################
	    
	    return (x,res_p,res_d,i)




	#####################################################################################
	softthresh = lambda x,th : np.sign(x)*np.maximum(np.abs(x)-th,0)


	def soft(self,x,T):
	#        y = soft(x,T)
	# soft-thresholding function
	## proximity operator for l1 norm
	    if np.sum(abs(T.flatten()))==0:
	       y = x.copy();
	    else:
	       y = np.maximum(abs(x) - T, 0);
	       y = y/(y+T) * x;
	    return y
	#def soft(x,T):
	#    T = T + np.spacing(1);
	#    y = (np.abs(x) - T).max(1);
	#    y_n=(np.divide(y,(y+T.T)))
	#    y = y_n.T*x;
	#    return y
	    
	def soft_neg(y,TAU):
	    z = np.max(np.abs(y+TAU/2) - TAU/2, 0);
	    z = z/(z+TAU/2) * (y+TAU/2);
	    return z

	def hinge(y):
	    z = np.max(-y,0);
	    return z