import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.cosmology as cosmology
import astropy.units as u
import astropy.constants as const
from cmath import sqrt as csqrt


### CONSTANTS ###
c = const.c
G = const.G
H0 = cosmology.Planck15.H0
RHO_CRIT = 3 * H0**2 / (8 * np.pi * G) # only use rho_crit at current time
u.set_enabled_equivalencies(u.dimensionless_angles()) # allow angles to be dimensionless

def SIGMA_CRIT(DS, DL):
    # Critical surface density for lens at distance DL and source at distance DS
    zS = cosmology.z_at_value(cosmology.Planck15.angular_diameter_distance, DS, zmax=1.1)
    zL = cosmology.z_at_value(cosmology.Planck15.angular_diameter_distance, DL, zmax=1.1)
    DSL = cosmology.Planck15.angular_diameter_distance_z1z2(zL, zS)
    return c**2 / (4 * np.pi * G) * (DS / (DSL * DL))


### GALAXY/HALO OBJECTS ###

class Halo:

    def __init__(self, M200, DL):
        self.M200 = M200
        self.DL = DL

    def shear(self, T, DS):
        # Tangential shear at angle T for object at distance DS
        return ((self.averageSurfaceDensity(T) - self.surfaceDensity(T)) / SIGMA_CRIT(DS, self.DL)).to_value("")

    def ellipticity(self, T, DS):
        # Magnitude of ellipticity at angle T for object at distance DS
        convergence = self.surfaceDensity(T) / SIGMA_CRIT(DS, self.DL)
        reducedShear = self.shear(T, DS) / (1 - convergence)
        return (2 * reducedShear / (1 + reducedShear**2)).to_value("")

    def deflection(self, Bx, By, DS):
        # Gives the image angle theta (T) that an object at source angle beta (B) and distance DS is deflected to
        return Bx, By

    def lense(self, galaxy):
        # deflection angle: beta -> theta
        Tx, Ty = self.deflection(galaxy.Bx, galaxy.By, galaxy.DS)
        T = np.sqrt(Tx**2 + Ty**2)
        phi = np.arctan2(Ty, Tx)
        e = self.ellipticity(T, galaxy.DS)
        e1 = galaxy.e1 - e*np.cos(2*phi)
        e2 = galaxy.e2 - e*np.sin(2*phi)
        return LensedBackgroundGalaxy(Tx, Ty, e1, e2, galaxy.DS)


class NFWHalo(Halo):

    def __init__(self, M200, C, DL):
        super().__init__(M200, DL)
        self.C = C
        self.delta_c = (200/3) * C**3 / (np.log(1+C) - C/(1+C))
        r200 = ((3 * M200) / (800 * np.pi * RHO_CRIT))**(1/3)
        self.rs = r200/C
        self.Ts = self.rs/DL * u.rad

    def surfaceDensity(self, T):
        # Surface density at angle T
        x = T/self.Ts
        sigma = 2*RHO_CRIT * self.delta_c * self.rs / (x**2 - 1) * (1 - 2 / csqrt(x**2 - 1) * np.arctan(csqrt((x-1)/(x+1))))
        if sigma.imag != 0: print(sigma.imag)
        return sigma.real

    def averageSurfaceDensity(self, T):
        # Average surface density at angle T
        x = T/self.Ts
        sigma_bar = 4*RHO_CRIT * self.delta_c * self.rs / x**2 * (2 / csqrt(x**2 - 1) * np.arctan(csqrt((x-1)/(x+1))) + np.log(x/2))
        if sigma_bar.imag != 0: print(sigma_bar.imag)
        return sigma_bar.real


class IsothermalHalo(Halo):

    def __init__(self, M200, rc, DL):
        super().__init__(M200, DL)
        self.rc = rc
        self.Tc = rc/DL * u.rad
        r200 = ((3 * M200) / (800 * np.pi * RHO_CRIT))**(1/3)
        self.sigmaSquared = M200 * G / (2 * (r200 - rc * np.arctan(r200/rc)))
        self.T0 = lambda DS: 4*np.pi*self.sigmaSquared / c**2 * (DS - self.DL)/(DS)

    def surfaceDensity(self, T):
        # Surface density at angle T
        return self.sigmaSquared / (2 * G * self.DL * np.sqrt(T**2 + self.Tc**2))

    def averageSurfaceDensity(self, T):
        # Average surface density at angle T
        return self.sigmaSquared * (np.sqrt(T**2 + self.Tc**2) - self.Tc) / (G * self.DL * T**2)


class BackgroundGalaxy:

    def __init__(self, Bx, By, e1, e2, DS):
        self.Bx = Bx
        self.By = By
        self.e1 = e1
        self.e2 = e2
        self.DS = DS


class LensedBackgroundGalaxy:

    def __init__(self, Tx, Ty, e1, e2, DS):
        self.Tx = Tx
        self.Ty = Ty
        self.e1 = e1
        self.e2 = e2
        self.DS = DS
        self.T = np.sqrt(Tx**2 +Ty**2)
        self.phi = np.arctan2(Ty, Tx)
