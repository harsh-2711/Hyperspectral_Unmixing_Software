# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:02:22 2019

@author: ross
"""
import numpy as np
import scipy as sp
from Modules.Linear_Unmixing import LMM as lmm


def golden_section( theta0,df,MPlus,y,lb,ub):
#------------------------------------------------------------------
# This function allows ones to determine the optimal constrained line 
# search parameter using the golden section method.
# 
# 
# INPUT
#       theta0          : current point estimate
#       df              : current derivative vector
#       MPlus           : the endmember matrix
#       y               : observation vector to be unmixed
#       lb              : lower bound of the line search parameter
#       ub              : upper bound of the line search parameter
#
# OUTPUT
#       val             : optimal value of the line search parameter
#
#------------------------------------------------------------------

    L,R=MPlus.shape;
    b=[];a=[];lammbda=[];mu=[];
    l=10**(-6);
    alpha=0.618;
    b.append(ub);
    a.append(lb);
    lammbda.append(lb+(1-alpha)*(ub-lb));
    mu.append(lb+alpha*(ub-lb));
    al=np.zeros(R)
    al[:R-1]=np.squeeze(theta0[:R-1])-lammbda[0]*df[:R-1];
    al[R-1]=1-sum(al[:R-1]);
    for r in range(R,R*(R+1)/2):
        al=np.append(al,theta0[r]-lammbda[0]*df[r-1]);
    
    err_lambda = np.linalg.norm(y-gene_Gamma(MPlus,al[:,np.newaxis]))**2 ;
    
    al=np.zeros(R)
    al[:R-1]=np.squeeze(theta0[:R-1])-mu[0]*df[:R-1];
    al[R-1]=1-sum(al[:R-1]);
    for r in range(R,R*(R+1)/2):
        al=np.append(al,theta0[r]-mu[0]*df[r-1]);
    
    err_mu = np.linalg.norm(y-gene_Gamma(MPlus,al[:,np.newaxis]))**2 ;
    
    k=0;
    f=0;
    while((b[k]-a[k]) > l):
            
        if(err_lambda>err_mu and f==0):
            a.append(lammbda[k]);
            b.append(b[k]);
            lammbda.append(mu[k]); 
            mu.append(a[k+1]+alpha*(b[k+1]-a[k+1]));
            al=np.zeros(R)
            al[:R-1]=np.squeeze(theta0[:R-1])-mu[k+1]*df[:R-1]; 
            al[R-1]=1-np.sum(al[:R-1]);
            for r in range(R,R*(R+1)/2):
                al=np.append(al,theta0[r]-mu[k+1]*df[r-1]);
            
            err_lambda=err_mu;
            err_mu = np.linalg.norm(y-gene_Gamma(MPlus,al[:,np.newaxis]))**2;
            f=1;
            
        elif(err_lambda <= err_mu and f==0):
            a.append(a[k]);
            b.append(mu[k]);
            mu.append(lammbda[k]); 
            lammbda.append(a[k+1]+(1-alpha)*(b[k+1]-a[k+1]));
            al=np.zeros(R)
            al[:R-1]=np.squeeze(theta0[:R-1])-lammbda[k+1]*df[:R-1]; 
            al[R-1]=1-np.sum(al[:R-1]);
            for r in range(R,R*(R+1)/2):
                al=np.append(al,theta0[r]-lammbda[k+1]*df[r-1]);
            
            err_mu=err_lambda;
            err_lambda = np.linalg.norm(y-gene_Gamma(MPlus,al[:,np.newaxis]))**2;
            f=1;
        
        k=k+1;
        f=0;
    
    val = a[k]+(b[k]-a[k])/2;
    return val
    
def gene_Gamma(MPlus,theta0):
    R=MPlus.shape[1];
    a=theta0[:R];
    u=1;
    y=MPlus.dot(a);
    
    for i in range(R-1):
        for j in range(i+1,R):
            y=y+ theta0[R-1+u]*a[i]*a[j]*MPlus[:,[i]]*MPlus[:,[j]];
            u=u+1;
    return y

def GBM_gradient(y,MPlus):

#------------------------------------------------------------------
# NONLINEAR SPECTRAL UNMIXING ASSUMING THE GENERALIZED BILINEAR MODEL
# by A. HALIMI, Y. ALTMANN, N. DOBIGEON and J.-Y. TOURNERET 
# IRIT/ENSEEIHT/TéSA - France - 2011
# 
# INPUT
#       y        : L x 1 pixel to be unmixed where L is the number 
#                  of spectral band
#       MPlus    : L x R endmember matrix involved in the mixture
#                  where R is the number of endmembers
#
# OUTPUT
#       alpha    : abundance vector estimated by the procedure
#       gam      : gamma vector estimated by the procedure
#
#------------------------------------------------------------------

# number of spectral bands x number of endmember spectra
    L,R=MPlus.shape;
    
    
    ## Initialization
    Niter=900;
    # Initialization with the FCLS algorithm (assuming the linear mixing model) 
    # for the abundance vector
    fcls=lmm.FCLS(y.T,MPlus.T).T
    theta0=np.hstack((fcls.T,0.01*np.ones((1,R*(R-1)/2)))).T;
    
    THETA=np.zeros((theta0.shape[0],Niter))
    ER=np.zeros((Niter))
    DALPHA=np.zeros((Niter))
    DER=np.zeros((Niter))
    THETA[:,[0]]=theta0;
#    THETA=np.append(THETA,theta0)[:,np.newaxis];
    ER[0]=20;
    Gam_sq=0.01*np.ones((R,R))
    
    ## Iterations
    for t in range(1,Niter):
        y0=gene_Gamma(MPlus,theta0);
    
        # Derivatives with respect to the R-1 first abundances
        d_alpha=np.zeros((MPlus.shape[0],R-1))
        for r in range(R-1):
            d_alpha[:,[r]]= MPlus[:,[r]]-MPlus[:,[R-1]];
            for i in range(R-1):
                if i!=r:
                   d_alpha[:,[r]] = d_alpha[:,[r]] + Gam_sq[r,i]*theta0[i]*MPlus[:,[r]]*MPlus[:,[i]] - Gam_sq[r,R-1]*theta0[i]*MPlus[:,[r]]*MPlus[:,[R-1]]- Gam_sq[i,R-1]*theta0[i]*MPlus[:,[i]]*MPlus[:,[R-1]];
                
            d_alpha[:,[r]] = d_alpha[:,[r]] + Gam_sq[r,R-1]*MPlus[:,[r]]*MPlus[:,[R-1]] - 2* Gam_sq[r,R-1]*theta0[r]*MPlus[:,[r]]*MPlus[:,[R-1]];     
          
        
        # Derivative with respect to gamma 
        d_gamma=np.zeros((L,R,R));
        df_gamma=np.zeros((R-1,R))
        for i in range(R-1):
            for j in range(i+1,R):
                d_gamma[:,i,j]=np.squeeze(theta0[i]*theta0[j]*(MPlus[:,[i]])*(MPlus[:,[j]]));
                df_gamma[i,j]=(y0-y).T.dot(np.squeeze(d_gamma[:,i,j])[:,np.newaxis]); 
            
        df_alpha=((y0-y).T).dot(d_alpha);
        
        # Modification of the gradient vector for move optimization
        if sum(theta0[:R-1])>0.99:
            n=np.ones((R-1,1));
            if (df_alpha*n<0):
                df_alpha[R-1]=-sum(df_alpha[0,:R-2]);
            
        for r in range(R-1):
            if theta0[r]< 0.01:
                if df_alpha[r]>0:
                    df_alpha[r]=0;
              
        # Determination of the bounds for the line search parameter
        if(df_alpha==np.zeros((1,R-1))).all():
            lb=0;
            ub=0;
        else:
            lb=min(theta0[0]/df_alpha[:,0]*(df_alpha[:,0]!=0),(theta0[0]-1)/df_alpha[:,0])*(df_alpha[:,0]!=0);
            ub=max(theta0[0]/df_alpha[:,0]*(df_alpha[:,0]!=0),(theta0[0]-1)/df_alpha[:,0])*(df_alpha[:,0]!=0);
            for r in range(R-1):
                lb=max(lb,min(theta0[r]/df_alpha[:,r]*(df_alpha[:,r]!=0),(theta0[r]-1)/df_alpha[:,r])*(df_alpha[:,r]!=0));
                ub=min(ub,max(theta0[r]/df_alpha[:,r]*(df_alpha[:,r]!=0),(theta0[r]-1)/df_alpha[:,r])*(df_alpha[:,r]!=0));
                for j in range(r+1,R):
                   lb=max(lb, min(Gam_sq[r,j]/df_gamma[r,j]*(df_gamma[r,j]!=0),(Gam_sq[r,j]-1)/df_gamma[r,j])*(df_gamma[r,j]!=0));
                   ub=min(ub, max(Gam_sq[r,j]/df_gamma[r,j]*(df_gamma[r,j]!=0),(Gam_sq[r,j]-1)/df_gamma[r,j])*(df_gamma[r,j]!=0));
              
        lb=max(lb,min(np.sum(theta0[:R-1])/np.sum(df_alpha[:,:R-1])*(np.sum(df_alpha[:,:R-1])!=0),(np.sum(theta0[:R-1])-1)/np.sum(df_alpha[:,:R-1]))*(np.sum(df_alpha[:,:R-1])!=0));
        ub=min(ub,max(np.sum(theta0[:R-1])/np.sum(df_alpha[:,:R-1])*(np.sum(df_alpha[:,:R-1])!=0),(np.sum(theta0[:R-1])-1)/np.sum(df_alpha[:,:R-1]))*(np.sum(df_alpha[:,:R-1])!=0));  
        
        lb = max(lb,0);
        ub = max(ub,0);
        df=df_alpha.copy();
        u=1;
        for i in range(R-1):
            for j in range(i+1,R):
                df=np.append(df,df_gamma[i,j]);
                u=u+1;
            
        # Line search procedure using the Golden section method
        lambda1=golden_section( theta0.copy(),df.copy(),MPlus.copy(),y.copy(),lb,ub);
        
        # Unknown parameters update
        theta0[:R-1]=(np.squeeze(theta0[:R-1])-lambda1*df[:R-1])[:,np.newaxis];
        theta0[R-1]=1-np.sum(theta0[:R-1]);
        
        for r in range(R,R*(R+1)/2):
            theta0[r]=theta0[r]-lambda1*df[r-1];
        
        THETA[:,[t]]=theta0;
        DALPHA[t]=np.linalg.norm(THETA[:,t]-THETA[:,t-1]);
        ER[t]=np.linalg.norm(y-gene_Gamma(MPlus,THETA[:,[t]]))**2;
        DER[t]=ER[t]-ER[t-1];
        u=1;
        for i in range(R-1):
            for j in range(i+1,R):
                Gam_sq[i,j]=theta0[R-1+u];
                u=u+1;
                Gam_sq[j,i]=Gam_sq[i,j];
            
        if(abs(DER[t]) < 10**(-5)):
            break;
        alpha=theta0[:R];
        gam=theta0[R:R*(R+1)/2];
    
    return alpha,gam

#import scipy as sp
#y=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/GBM_unmix_GDA/y.mat")['y']
#MPlus=sp.io.loadmat(r"/home/ross/Codes/ReferenceCodes/chang_li/GBM_unmix_GDA/MPlus")['MPlus']
#alpha0=np.array([0.3,0.6,0.1]).T; # Actual abundance vector
#gamma0=np.array([0.5, 0.1, 0.3]).T; # Actual gamma vector
#sigma20= 3*10**-4; # Actual noise variance  
#alpha,gam = GBM_gradient(y, MPlus);
#print alpha0
#print alpha
