#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  a template for a simple galaxy formation model a la Krumholz & Dekel (2012); 
#                    see also Feldmann 2013
#
#   Andrey Kravtsov, 2019
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from colossus.cosmology import cosmology
from scipy.interpolate import UnivariateSpline

def fg_in(Mh):
    return 1.0

def R_loss(dt):
    """
    fraction of mass formed in stars that is returned back to the ISM
    """
    return 0.46

class model_galaxy(object):

    def __init__(self,  t = None, Mh = None, Mg = None, Ms = None, MZ = None, Z_IGM = 1.e-4, sfrmodel = None, tausf=2., cosmo = None, verbose = False):

        self.Zsun = 0.02

        if cosmo is not None: 
            self.cosmo = cosmo
            self.fbuni = cosmo.Ob0/cosmo.Om0
        else:
            errmsg = 'to initialize gal object it is mandatory to supply the collossus cosmo(logy) object!'
            raise Exception(errmsg)
            return
            
        if Mh is not None: 
            self.Mh = Mh
        else:
            errmsg = 'to initialize gal object it is mandatory to supply Mh!'
            raise Exception(errmsg)
            return
            
        if t is not None: 
            self.t = t # in Gyrs
            self.z = self.cosmo.age(t, inverse=True)        
            self.gr = self.cosmo.growthFactor(self.z)
            self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
            self.thubble = self.cosmo.hubbleTime(self.z)
            self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        else:
            errmsg = 'to initialize gal object it is mandatory to supply t!'
            raise Exception(errmsg)
            return
            
        # metallicity yield of a stellar population - this is a constant derived from SN and AGB simulations
        self.yZ = 0.069; 
        # assumed metallicity of freshly accreting gas (i.e. metallicity of intergalactic medium)
        self.Z_IGM = Z_IGM; 
        
        if Ms is not None:
            self.Ms = Ms
        else: 
            self.Ms = 0.0
        if Mg is not None:
            self.Mg = Mg
        else: 
            self.Mg = self.fbuni*Mh
            
        if MZ is not None:
            self.MZ = MZ
        else: 
            self.MZ = self.Z_IGM*self.Mg
        if MZ is not None and Mg is not None:
            # model for molecular hydrogen content is to be implemented here
            self.MH2 = 0.0
        else:
            self.MH2 = 0.0
        
        # only one model based on total gas density for starters, a better model is to be implemented
        #self.sfr_models = {'gaslinear': self.SFRgaslinear, 'H2linear': self.SFRlinear}
        #if not hasattr(self, 'sfr_models'):
        sfr_models = getattr(self, 'sfr_models', None)
        if sfr_models is None:
            self.sfr_models = {'gaslinear': self.SFRgaslinear}
    
        if not hasattr(self, 'sfrmodel'):
            try: 
                self.sfr_models[sfrmodel]
            except KeyError:
                print("unrecognized sfrmodel in model_galaxy.__init__:", sfrmodel)
                print("available models:", self.sfr_models)
                return
            self.sfrmodel = sfrmodel
        else:
            if self.sfrmodel is None:
                errmsg = 'to initialize gal object it is mandatory to supply sfrmodel!'
                raise Exception(errmsg)
            return
            
        if self.sfrmodel is 'gaslinear':
            self.tausf = tausf
            
        if verbose is not None:
            self.verbose = verbose
        else:
            errmsg = 'verbose parameter is not initialized'
            raise Exception(errmsg)
            return

        self.epsout = self.eps_out(); self.Mgin = self.Mg_in(t); 
        self.Msin = self.Ms_in(t)
        self.sfr = self.SFR(t)
        self.Rloss = R_loss(0.); self.Rloss1 = 1.0-self.Rloss

        return
        
    
    def dMhdt(self, Mcurrent, t):
        """
        halo mass accretion rate using approximation of eqs 3-4 of Krumholz & Dekel 2012
        output: total mass accretion rate in Msun /Gyr
        """
        self.Mh = Mcurrent
        # this equation is eq. 1 from Feldmann 2013
        dummy = 1.06e12*(Mcurrent/1.e12)**1.14 *self.dDdt/(self.gr*self.gr)

        # approximation in Krumholz & Dekel (2012) for testing 
        #dummy = 5.02e10*(Mcurrent/1.e12)**1.14*(1.+self.z+0.093/(1.+self.z)**1.22)**2.5    

        return dummy
                
    def eps_in(self, t):
        """
        fraction of universal baryon fraction that makes it into galaxy 
        along with da
        """
        epsin = 1.0
        return epsin

    def Mg_in(self, t):
        dummy = self.fbuni*self.eps_in(t)*fg_in(self.Mh)*self.dMhdt(self.Mh,t)
        return dummy
    
    def Ms_in(self, t):
        dummy = self.fbuni*(1.0-fg_in(self.Mh))*self.dMhdt(self.Mh,t)
        return dummy

    def tau_sf(self):
        """
        gas consumption time in Gyrs 
        """
        return self.tausf

    def SFRgaslinear(self, t):
        return self.Mg/self.tau_sf()
         
    def SFR(self, t):
        """
        master routine for SFR - 
        eventually can realize more star formation models
        """  
        return self.sfr_models[self.sfrmodel](t)
        
    def dMsdt(self, Mcurrent, t):
        dummy = self.Msin + self.Rloss1*self.sfr
        return dummy

    def eps_out(self):
        return 0.0
        
    def dMgdt(self, Mcurrent, t):
        dummy = self.Mgin - (self.Rloss1 + self.epsout)*self.sfr
        return dummy

    def zeta(self):
        """
        output: fraction of newly produced metals removed by SNe in outflows
        """
        return 0.0

    def dMZdt(self, Mcurrent, t):
        dummy = self.Z_IGM*self.Mgin + (self.yZ*self.Rloss1*(1.-self.zeta()) - (self.Rloss1+self.epsout)*self.MZ/(self.Mg))*self.sfr
        return dummy
        
    def evolve(self, Mcurrent, t):
        # first set auxiliary quantities and current masses
        self.z = self.cosmo.age(t, inverse=True)        
        self.gr = self.cosmo.growthFactor(self.z)
        self.dDdz = self.cosmo.growthFactor(self.z,derivative=1)
        self.thubble = self.cosmo.hubbleTime(self.z)
        self.dDdt = self.cosmo.growthFactor(self.z, derivative=1) * self.cosmo.age(t, derivative=1, inverse=True)

        self.Mh = Mcurrent[0]; self.Mg = Mcurrent[1]; 
        self.Ms = Mcurrent[2]; self.MZ = Mcurrent[3]
        self.epsout = self.eps_out(); self.Mgin = self.Mg_in(t); 
        self.Msin = self.Ms_in(t)
        self.Rloss = R_loss(0.); self.Rloss1 = 1.0-self.Rloss
        self.sfr = self.SFR(t)
        
        # calculate rates for halo mass, gas mass, stellar mass, and mass of metals
        dMhdtd = self.dMhdt(Mcurrent[0], t)
        dMgdtd = self.dMgdt(Mcurrent[1], t)
        dMsdtd = self.dMsdt(Mcurrent[2], t)
        dMZdtd = self.dMZdt(Mcurrent[3], t)
        
        if self.verbose:
            print("evolution: t=%2.3f Mh=%.2e, Mg=%.2e, Ms=%.2e, Z/Zsun=%2.2f,SFR=%4.2e"%(t,self.Mh,self.Mg,self.Ms,self.MZ/self.Mg/0.02,self.SFR(t)*1.e-9))

        return [dMhdtd, dMgdtd, dMsdtd, dMZdtd]


class model_uv_heating(model_galaxy):
    def __init__(self, *args, **kwargs):
        super(model_uv_heating, self).__init__(*args, **kwargs)
        return
        
    def UV_cutoff(self, z):
        """
        approximation to the cutoff mass in Fig 3 of Okamoto, Gao & Theuns 2008
        the output is mass in /h Msun. 
        """
        dummy = np.zeros_like(z)
        dummy[z>9] = 1.e6
        dummy[z<=9] = 6.e9*np.exp(-0.63*z[z<9]) # expression from Nick
        return  1.0/(1.0+(2.**(2./3.)-1.)*(dummy/self.Mh)**2)**(1.5)

    def Mg_in(self, t):
        dummy = self.fbuni*self.fg_in(t)*self.eps_in(t)*self.dMhdt(self.Mh,t)
        return dummy

    def Ms_in(self, t):
        dummy = 0.0
        return dummy

    def fg_in(self,t):
        return self.UV_cutoff(self.z)
    
    def eps_in(self, t):
        zd = self.cosmo.age(t, inverse=True)
        epsin = 1.0
        return epsin

class gmodel_UVheating(model_galaxy):
    def __init__(self, *args, **kwargs):

        super(gmodel_UVheating, self).__init__(*args, **kwargs)
        return
       
    def UV_cutoff(self, z):
        """
        approximation to the cutoff mass in Fig 3 of Okamoto, Gao & Theuns 2008
        the output is mass in /h Msun. 
        """
        dummy = np.zeros_like(z)
        dummy[z>9] = 1.e6
        dummy[z<=9] = 6.e9*np.exp(-0.63*z[z<9]) # expression from Nick
        return  1.0/(1.0+(2.**(2./3.)-1.)*(dummy/self.Mh)**2)**(1.5)

    def Mg_in(self, t):
        dummy = self.fbuni*self.fg_in(t)*self.eps_in(t)*self.dMhdt(self.Mh,t)
        return dummy

    def Ms_in(self, t):
        dummy = 0.0
        return dummy

    def fg_in(self,t):
        zd = self.cosmo.age(t, inverse=True)
        # here I implement soft suppression of baryon fraction, as seen in simulation (eq 2.2 in the notes)
        # results are qualitatively the same, but this function makes suppression a "softer" as a function of M
        return self.UV_cutoff(zd)
    
    def eps_in(self, t):
        return 1.0


from colossus.halo.concentration import concentration
from colossus.halo.mass_defs import changeMassDefinition


class gmodel_r50(gmodel_UVheating):

    def __init__(self, *args, **kwargs):

        # process kwargs and remove (via kwargs.pop extraction) those
        # keyword arguments that should not be passed to parent init
        if 'ac' in kwargs: # Flag to use adiabatic contraction correction
            self.ac = kwargs.pop('ac')
        else:
            self.ac = 'off'
        # factor in front of the MMW98 expression for Rd to set the fraction
        # of angular momentum lost during galaxy evolution
        if 'etar' in kwargs:
            self.etar = kwargs.pop('etar')
        else:
            self.etar = 1.0
                                                
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False
            
        if 'tausf' in kwargs:
            self.tausf = kwargs.pop('tausf')
        else:
            self.tausf = 2.0
            
        if 'sfrmodel' in kwargs:
            self.sfr_models = {'gaslinear': self.SFRgaslinear}
            try: 
                self.sfr_models[kwargs['sfrmodel']]
            except KeyError:
                print("unrecognized sfrmodel in model_galaxy.__init__:", kwargs['sfrmodel'])
                print("available models:", self.sfr_models)
                return
            self.sfrmodel = kwargs['sfrmodel']
        elif self.sfrmodel is None:
            errmsg = 'to initialize gal object it is mandatory to supply sfrmodel!'
            raise Exception(errmsg)
            return
                
        # initialize everything in the parent class
        super(gmodel_r50, self).__init__(*args, **kwargs)
        
        return

    def eta50_MMW98(self, c, jmlam, md):
        """
        compute the factor between r50 and R200c in the MMW98 model
        input: c=concentration; jmlam = (j_d/m_d)*lambda; md = M*/M200c
        """
        # see eq. 23 in Mo, Mao & White 1998
        fc = 0.5*c*(1.-1./(1.+c)**2-2.*np.log(1.+c)/(1.+c))/(c/(1.+c)-np.log(1.+c))**2
        # use the expression below if you want to include full MMW98 model that includes adiabatic contraction
        if self.ac == 'off':
            fR = 1.0
        else:
            fR = (jmlam/0.1)**(-0.06+2.71*md+0.0047/jmlam)*(1.-3.*md+5.2*md**2)*(1.-0.019*c+2.5e-4*c*c+0.52/c)
        eta = 1.187/np.sqrt(fc)*jmlam*fR
        return eta

    def r50(self, t):
        # Rdisk using formula from Mo, Mao & White 1998
        Mhh = self.Mh * self.cosmo.h
        cvir = concentration(Mhh, 'vir', self.z, model='diemer15')
        M200c, R200c, c200c = changeMassDefinition(Mhh, cvir, self.z, 'vir', '200c')
        md = self.Ms/(M200c/self.cosmo.h); 
        jmlam = self.etar*0.045 # assume etar fraction of the angular momentum lost
        eta50 = self.eta50_MMW98(c200c, jmlam, md)
        r_50 = eta50 * R200c / self.cosmo.h
        return r_50 
    
    def R_d(self, t):
        rd = self.r50(t)/1.687
        return rd 
     
    # redefine star formation function to use tau_sf that can be set on input
    def SFRgaslinear(self, t):
        return self.Mg/self.tausf


class gmodel_H2(gmodel_r50):

    def __init__(self, *args, **kwargs):

        if 'sfrmodel' in kwargs:
            self.sfr_models = {'gaslinear': self.SFRgaslinear, 'H2linear': self.SFRH2linear}
            try: 
                self.sfr_models[kwargs['sfrmodel']]
            except KeyError:
                print("unrecognized sfrmodel in model_galaxy.__init__:", kwargs['sfrmodel'])
                print("available models:", self.sfr_models)
                return
            self.sfrmodel = kwargs.pop('sfrmodel')
        elif self.sfrmodel is None:
            errmsg = 'to initialize gal object it is mandatory to supply sfrmodel!'
            raise Exception(errmsg)
            return
        # initialize everything in the parent class        
        super(gmodel_H2, self).__init__(*args, **kwargs)
        
        return
            
    def Sigma_gas(self, r, t):
        rd = self.R_d(t)
        Sigma0 = self.Mg/(2.*np.pi*rd**2)/self.cosmo.h
        return Sigma0*np.exp(-r/rd)
    
    def f_H2(self, Sigmag, ZZsun):
        """
        Krumholz et al. model for H2 
        Sigmag = surface density of gas on ~kpc scale in Msun/kpc^2
            ZZsun = gas metallicity in units of solar 
        """
        ZdZsun = np.maximum(ZZsun, 0.08)
        x = 3.1/4.1*(1.+3.1*ZdZsun**0.365)
        tc = 3.34e-7* Sigmag * ZdZsun  # assumes clumpiness factor of c=5
        s = np.log(1.+0.6*x + 0.01*x*x)/(0.6*tc) 
        isl2 = (s<2)
        dummy = np.zeros_like(Sigmag)
        dummy[isl2] = (1.0 - 0.75*s[isl2]/(1.+0.25*s[isl2]))
        return dummy
   
    def M_H2(self,t):
        ZZsun = self.MZ/self.Mg/self.Zsun
        rd = self.R_d(t)
        rg = np.linspace(0., 20.*rd, 50) 
        Sigma0 = self.Mg/(2.*np.pi*rd**2)
        sgd = Sigma0*np.exp(-rg/rd)
        sgd = rg * sgd * self.f_H2(sgd, ZZsun) 
        sgsp = UnivariateSpline(rg,sgd,s=0.0)
        dummy = 2.0 * np.pi * sgsp.integral(0., rg[-1])
        return dummy
          
    def SFRH2linear(self, t):
        """
        linear star formation rate based on MH2
        input: t = time in Gyr
        """
        return self.M_H2(t)/self.tausf

class gmodel_heating(gmodel_H2):
    def __init__(self, *args, **kwargs):

        super(gmodel_heating, self).__init__(*args, **kwargs)
        return

    def Mhot(self,z):
        # the factor in front is adjusted to reproduce results of actual calculations with tcool
        # the redshift scaling follows from tcool(z)~t_age(z) requirement
        mhot = 3.e11 * (self.cosmo.rho_c(z)/ self.cosmo.rho_c(0))**(0.25)
        return mhot
           
    def eps_in(self, t):
        zd = self.cosmo.age(t, inverse=True)
        alfa = 0.5
        epsin = 1.0 - 1.0/(1.0+(2.**(alfa/3.)-1.)*(self.Mhot(zd)/self.Mh)**alfa)**(3./alfa)
        return epsin

class gmodel_full(gmodel_heating):

    def __init__(self, *args, **kwargs):

        self.wind_models = {'powerlaw': self.Muratov15wind, 
                            'energywind': self.energywind, 
                            'Muratov15mod': self.Muratov15modified}
        if 'windmodel' in kwargs: 
            try: 
                windmodel = kwargs.pop('windmodel')
                self.wind_models[windmodel]
            except KeyError:
                print("unrecognized windmodel in model_galaxy.__init__:", windmodel)
                print("available models:", self.wind_models)
                return
            self.windmodel = windmodel
            # mass loading factor is parameterized as eta = etawind x (M*/10^{10} Msun)^{alfawind}
            # for Muratov et al. 2015 calibration: etawind=3.6, alfawind = -0.35
            # etawind = 0 corresponds to no wind
            self.etawind = kwargs.pop('etawind')
            self.alfawind = kwargs.pop('alfawind')
               
        else:
            errmsg = 'to initialize gal object it is mandatory to supply windmodel!'
            raise Exception(errmsg)
            return
                                
        # initialize everything in the parent class
        super(gmodel_full, self).__init__(*args, **kwargs)
        
        return

    def Muratov15wind(self):
        if self.Ms > 0.:
            return self.etawind*(self.Ms/1.e10)**(self.alfawind)
        else:
            return 0.

    def energywind(self):
        """
        model implementing eta ~ Mhalo^{-2/3} scaling expected for energy-driven wind
        """
        return np.maximum(0., self.etawind*(self.Mh/5.e11)**(self.alfawind) - 4.6)

    def Muratov15modified(self):
        if self.Ms > 0: 
            return np.maximum(0.,self.etawind*(self.Ms/1.e10)**(self.alfawind)-4.6)
        else:
            return 0.

    def eps_out(self):
        dummy = self.wind_models[self.windmodel]()
        return dummy