import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.cosmology as cosmology
from astropy.visualization import astropy_mpl_style
import astropy.units as u
import astropy.constants as const
from cmath import sqrt as csqrt

plt.style.use(astropy_mpl_style)
plt.rcParams["figure.dpi"] = 100

### CONSTANTS ###
c = const.c
G = const.G
H0 = cosmology.default_cosmology.get().H(0)
RHO_CRIT = 3 * H0**2 / (8 * np.pi * G) # only use rho_crit at current time
u.set_enabled_equivalencies(u.dimensionless_angles()) # allow angles to be dimensionless


### USEFUL FUNCTIONS ###

def SIGMA_CRIT(DS, DL):
    # Critical surface density for lens at distance DL and source at distance DS
    DSL = DS - DL
    return c**2 / (4 * np.pi * G) * (DS / (DSL * DL))

def lensingImage(halo, backgroundGalaxies, v, a):
    # Produce a mock lensing image with a halo and a population of background galaxies
    # v is the view width and a is the size of galaxies
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(-v/2,v/2)
    ax.set_ylim(-v/2,v/2)
    for gal in backgroundGalaxies:
        lensedGal = halo.lense(gal)
        lensedGalEllipse = lensedGal.ellipse(a)
        lensedGalEllipse.set_facecolor(np.random.rand(3))
        ax.add_artist(lensedGalEllipse)
    return plt.show()


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
        e1 = galaxy.e1 - e*np.cos(2*phi) # TODO: how do ellipticities add?
        e2 = galaxy.e2 - e*np.sin(2*phi)
        return LensedBackgroundGalaxy(Tx, Ty, e1, e2, galaxy.DS)

    def plotProperties(self, start, stop, step, DS):
        # Plot the shear, ellipticity, convergence, and magnification for the halo
        theta = np.linspace(start, stop, step)*u.arcsec
        epsilon = np.array([self.ellipticity(t, DS) for t in theta])
        gamma = np.array([self.shear(t, DS) for t in theta])
        kappa = np.array([(self.surfaceDensity(t) / SIGMA_CRIT(DS, self.DL)).to_value("") for t in theta])
        mu = 1/((1-kappa)**2 - gamma**2)
        plt.figure()
        plt.plot(theta, epsilon)
        plt.plot(theta, gamma)
        plt.plot(theta, kappa)
        plt.plot(theta, mu)
        plt.legend(["$\epsilon$", "$\gamma$", "$\kappa$", "$\mu$"])
        plt.xlabel("$\\theta$ (arcseconds)")
        plt.ylim(-10,10)
        return plt.show()



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
        self.T0 = lambda DS: 4*np.pi*halo_iso.sigmaSquared / c**2 * (DS - halo_iso.DL)/(DS)

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

    def ellipse(self, a):
        # Returns matplotlib ellipse object of approximate size a
        e = np.sqrt(self.e1**2 + self.e2**2)
        return Ellipse(xy=[self.Tx.to_value(u.arcsec),self.Ty.to_value(u.arcsec)],
                       width=a,
                       height=a*(e+1)/(e-1),
                       angle=self.phi.to_value(u.degree))



if __name__ == '__main__':
    # here is an example of lensing very close to the halo
    v = 100 # view radius

    halo_iso = IsothermalHalo(
        M200=1e15*u.solMass,
        rc=10*u.kpc,
        DL=1*u.Gpc)

    halo_nfw = NFWHalo(
        M200=1e15*u.solMass,
        C=10,
        DL=1*u.Gpc)

    halo_iso.plotProperties(0.0001, v, 500, 3*u.Gpc)
    print("theta_c:", halo_iso.Tc.to(u.arcsec))
    print("theta_0:", halo_iso.T0(3*u.Gpc).to(u.arcsec))

    halo_nfw.plotProperties(0.0001, v, 500, 3*u.Gpc)
    print("theta_s:", halo_nfw.Ts.to(u.arcsec))

    backgroundGalaxies = [BackgroundGalaxy(
        Bx=(v*np.random.rand()-v/2)*u.arcsec,
        By=(v*np.random.rand()-v/2)*u.arcsec,
        e1=0,
        e2=0,
        DS=3*u.Gpc) for i in range(1000)]

    lensingImage(halo_iso, backgroundGalaxies, v)
    lensingImage(halo_nfw, backgroundGalaxies, v)
