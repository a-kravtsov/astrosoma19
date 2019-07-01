#
#    auxiliary cosmology routines
#
#   
import numpy as np
from scipy.interpolate import UnivariateSpline
import sys
python_version = sys.version_info[0]

if python_version >=3: 
    from .auxiliary import romberg
else:
    from auxiliary import romberg


def d_func(a, **kwargs):
    """
    auxiliary function for the integrand of the comoving distance
    
    parameters:
    -----------
    a: float
       expansion factor of the epoch to which to compute comoving distance
    kwargs: keyword dictionary
        containing values of Om0, OmL, and Omk
    """
    a2i = 1./(a * a) 
    a2Hai = a2i / np.sqrt(kwargs["Om0"]/a**3 + kwargs["OmL"] + kwargs["Omk"] * a2i)
    return a2Hai
    
def dcom(z, Om0, OmL, ninter=20):
    """
    function computing comoving distance Dc for a given redshift and mean matter and vacuum energies Om0 and OmL
    """
    Omk = 1. - Om0 - OmL
    kwargs = {"Om0": Om0, "OmL": OmL, "Omk": Omk}
    a = 1. / (1.0 + z)
    
    nz = np.size(z)
    if nz == 1:
        if np.abs(a-1.0) < 1.e-10:
            dc = 0.
        else:
            dc = romberg(d_func, a, 1., rtol = 1.e-10, mmax = 16, verbose = False, **kwargs)[0]
    elif nz > 1:
        dc = np.zeros(nz)
        if nz <= ninter:
            for i, ad in enumerate(a):
                if np.abs(ad-1.0) < 1.e-10:
                    dc[i] = 0.
                else:
                    dc[i] = romberg(d_func, ad, 1., rtol = 1.e-10, mmax = 16, verbose = False, **kwargs)[0]
        else:
            zmin = np.min(z); zmax = np.max(z)
            zi = np.linspace(zmin, zmax, num=ninter)
            fi = np.zeros(ninter)
            for i, zid in enumerate(zi):
                aid = 1.0/(1+zid)
                if np.abs(aid-1.0) < 1.e-10:
                    fi[i] = 0.
                else:
                    fi[i] = romberg(d_func, aid, 1., rtol = 1.e-10, mmax = 16, verbose = False, **kwargs)[0]
            dsp = UnivariateSpline(zi, fi, s=0.)
            dc = dsp(z)
    return dc
    
def d_l(z, Om0, OmL, ninter=20):
    """
    function computing luminosity distance
    
    parameters:
    -----------
    z: float
        redshift
    Om0: mean density of matter in units of critical density at z=0
    OmL: density of vacuum energy in units of critical density
    nspline: integer
        number of spline nodes in redshift to use for interpolation if number of computed distances is > nsp
        
    returns:
    --------
    d_L in units of d_H=c/H0 (i.e. to get distance in Mpc multiply by 2997.92
    """
    
    Omk = 1. - Om0 - OmL
    zp1 = 1.0 + z
    dc = dcom(z, Om0, OmL, ninter)    
    if np.abs(Omk) < 1.e-15:
        return dc * zp1
    elif Omk > 0:
        sqrtOmk = np.sqrt(Omk)
        return np.sinh(sqrtOmk * dc) / sqrtOmk * zp1
    else:
        sqrtOmk = np.sqrt(-Omk)
        return np.sin(sqrtOmk * dc) / sqrtOmk * zp1
        
def d_a(z, Om0, OmL):
    zp1i = 1./(z + 1.)
    return d_l(z, Om0, OmL) * zp1i * zp1i


if __name__ == '__main__':
    z = 0.1; Om0 = 0.3; OmL = 0.7
    print("z = %.2f; Om0 = %.2f; Oml = %.2f:"%(z, Om0, OmL))
    print("d_L = %.3e; d_A = %.3e"%(d_l(z, Om0, OmL), d_a(z, Om0, OmL)))
