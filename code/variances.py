import numpy as np


factlnpi = np.log(10.)/(2.*np.pi**2)
kmin = 1.e-5; 
lkmin = np.log10(kmin); 

# first variances for the normal real-space top-hat filter

def dvar(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    return (3.*(np.sin(x)-x*np.cos(x))/(x**3))**2*k**3*Pka
        
def ddvar(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['r']; X = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    W_THx = 3.*(np.sin(x)-x*np.cos(x)) / x**3
    W_THX = 3.*(np.sin(X)-X*np.cos(X)) / X**3
    return W_THx*W_THX*k**3*Pka
        
def dddvar(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['r']; X = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    W_THx = 3.*(np.sin(x)-x*np.cos(x))/x**3
    W_THX = 3.*(X*np.sin(X)-3./X*(np.sin(X)-X*np.cos(X)))/X**2
    return W_THx*W_THX*k**3*Pka

def ddddvar(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['r']; X = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    W_THx = 3.*(x*np.sin(x)-3./x*(np.sin(x)-x*np.cos(x)))/x**2
    W_THX = 3.*(X*np.sin(X)-3./X*(np.sin(X)-X*np.cos(X)))/X**2
    return W_THx*W_THX*k**3*Pka

# now variances for Neal's smoothed top-hat filter 
a = 0.16; a2 = a*a
def dvars(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    return (3.*(np.sin(x)-x*np.cos(x))/(x**3))**2/(1.+(a*x)**2)**4*k**3*Pka
        
def ddvars(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['r']; X = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    W_THx = 3.*(np.sin(x)-x*np.cos(x)) / x**3 / (1.+(a*x)**2)**2
    W_THX = 3.*(np.sin(X)-X*np.cos(X)) / X**3 / (1.+(a*X)**2)**2
    return W_THx*W_THX*k**3*Pka
        
def dddvars(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['r']; X = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    W_THx = 3.*(np.sin(x)-x*np.cos(x)) / x**3 / (1.+a2*x**2)**2
    sinX = np.sin(X); sincosX = sinX - X*np.cos(X)
    a2X2p1 = (a2*X**2 + 1.)
    W_THX = 3.*sinX/X/a2X2p1**2 - 12.*a2*sincosX/X/a2X2p1**3 - 9.*sincosX/X**3/a2X2p1**2
    return W_THx*W_THX*k**3*Pka

def ddddvars(lk, **kwargs):
    k = 10.**lk; x = k * kwargs['r']; X = k * kwargs['R']
    cosmo = kwargs['cosmo']
    Pka = cosmo.matterPowerSpectrum(k)*np.exp(-(kwargs["Rc"]*k))
    sinx = np.sin(x); sincosx = sinx - x*np.cos(x)
    a2x2p1 = (a2*x**2 + 1.)
    W_THx = 3.*sinx/x/a2x2p1**2 - 12.*a2*sincosx/x/a2x2p1**3 - 9.*sincosx/x**3/a2x2p1**2
    sinX = np.sin(X); sincosX = sinX - X*np.cos(X)
    a2X2p1 = (a2*X**2 + 1.)
    W_THX = 3.*sinX/X/a2X2p1**2 - 12.*a2*sincosX/X/a2X2p1**3 - 9.*sincosX/X**3/a2X2p1**2
    return W_THx*W_THX*k**3*Pka


from .auxiliary import romberg

def delta_variance(r, R, rtol=1.e-8, func=None, cosmo=None):
    """
        function to compute Gaussian variance of delta on scale r
        and derivative d\delta/d\ln R on scale R
        func = function to integrate
    """
    rmin = np.minimum(r,R)
    if rmin <= 0.:
        raise Exception('input radius is <=0: %.3e %.3e'%(r,R))
        
    kmax = 1000.*2.*np.pi/rmin; lkmax = np.log10(kmax)
    Rc = 1.e-8*rmin
    kwargs = {'r': r, 'R': R, 'Rc': Rc, 'cosmo': cosmo}
    sig2dd, errlog = romberg(func, lkmin, lkmax,  mmax=12, rtol=rtol, **kwargs)
    sig2dd *= factlnpi
    return sig2dd
    
from scipy.special import erfc

def ex_func(x, **kwargs):
    Nr = kwargs['Nr']
    dummy = (1.-0.5*erfc(x))**(Nr-1)*x*np.exp(-x**2) 
    return dummy
   
factex = np.sqrt(2./np.pi)
def ex_mean(Nr, rtol=1.e-8):
    """
        function to compute mean extremum - the largest value of N random
        Gaussian numbers with zero mean and unit variance (see S 4 of Dalal et al. 2010)
        
    """
    xmin = -10.; xmax = 10.;

    kwargs = {'Nr': Nr}
    xexmean, errlog = romberg(ex_func, xmin, xmax, mmax=20, rtol=rtol, **kwargs)
    xexmean *= (factex * float(Nr))
    return xexmean
    
