import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.cosmology as cosmology
from astropy.visualization import astropy_mpl_style
import astropy.units as u
import astropy.constants as const

plt.style.use(astropy_mpl_style)

#Define constants
c = const.c
G = const.G
H0 = cosmology.default_cosmology.get().H(0)
RHO_CRIT = 3 * H0**2 / (8 * np.pi * G)
u.set_enabled_equivalencies(u.dimensionless_angles())

def SIGMA_CRIT(DS, DL):
    DSL = DS - DL
    return c**2 / (4 * np.pi * G) * (DS / (DSL * DL))


class NFWHalo:

    def __init__(self, M200, C, DL):
        self.M200 = M200
        self.C = C
        self.delta_c = (200/3) * C**3 / (np.log(1+C) - C/(1+C))
        r200 = ((3 * M200) / (800 * np.pi * RHO_CRIT))**(1/3)
        self.rs = r200/C
        self.DL = DL


class IsothermalHalo:

    def __init__(self, M200, rc, DL):
        self.M200 = M200
        self.rc = rc
        self.DL = DL
        self.Tc = rc/DL * u.rad
        r200 = ((3 * M200) / (800 * np.pi * RHO_CRIT))**(1/3)
        self.sigmaSquared = M200 * G / (2 * (r200 - rc * np.arctan(r200/rc)))

    def surfaceDensity(self, T):
        # Surface density at angle T
        return self.sigmaSquared / (2 * G * self.DL * np.sqrt(T**2 + self.Tc**2))

    def averageSurfaceDensity(self, T):
        # Average surface density at angle T
        return self.sigmaSquared * (np.sqrt(T**2 + self.Tc**2) - self.Tc) / (G * self.DL * T**2)

    def shear(self, T, DS):
        # Tangential shear at angle T for object at distance DS
        return (self.averageSurfaceDensity(T) - self.surfaceDensity(T)) / SIGMA_CRIT(DS, self.DL)

    def ellipticity(self, T, DS):
        # Magnitude of ellipticity at angle T for object at distance DS
        convergence = self.surfaceDensity(T) / SIGMA_CRIT(DS, self.DL)
        reducedShear = self.shear(T, DS) / (1 - convergence)
        return 2 * reducedShear / (1 + reducedShear**2)

    def deflection(self, Bx, By, DS):
        # Gives the image angle theta (T) that an object at source angle beta (B) and distance DS is deflected to
        return Bx, By

    def lense(self, galaxy):
        # deflection angle: beta -> theta
        Tx, Ty = self.deflection(galaxy.Bx, galaxy.By, galaxy.DS)
        T = np.sqrt(Tx**2 + Ty**2)
        phi = np.arctan2(Ty, Tx)
        e = self.ellipticity(T, galaxy.DS)
        e1 = -e*np.cos(2*phi)
        e2 = -e*np.sin(2*phi)
        return LensedBackgroundGalaxy(Tx, Ty, galaxy.a, e1, e2, galaxy.DS)


class BackgroundGalaxy:

    def __init__(self, Bx, By, a, e1, e2, DS):
        self.Bx = Bx
        self.By = By
        self.a = a
        self.e1 = e1
        self.e2 = e2
        self.DS = DS


class LensedBackgroundGalaxy:

    def __init__(self, Tx, Ty, a, e1, e2, DS):
        self.Tx = Tx
        self.Ty = Ty
        self.a = a
        self.e1 = e1
        self.e2 = e2
        self.DS = DS
        self.T = np.sqrt(Tx**2 +Ty**2)
        self.phi = np.arctan2(Ty, Tx)

    def ellipse(self):
        e = np.sqrt(self.e1**2 + self.e2**2)
        return Ellipse(xy=[self.Tx.to_value(u.arcsec),self.Ty.to_value(u.arcsec)],
                       width=self.a,
                       height=self.a*(e+1)/(e-1),
                       angle=self.phi.to_value(u.degree))


if __name__ == '__main__':
    v = 100

    halo = IsothermalHalo(
        M200=1e15*u.solMass,
        rc=10*u.kpc,
        DL=1*u.Gpc)

    backgroundGalaxies = [BackgroundGalaxy(
        Bx=(v*np.random.rand()-v/2)*u.arcsec,
        By=(v*np.random.rand()-v/2)*u.arcsec,
        e1=0,
        e2=0,
        a=1,
        DS=3*u.Gpc) for i in range(1000)]

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(-v/2,v/2)
    ax.set_ylim(-v/2,v/2)

    for gal in backgroundGalaxies:
        lensedGal = halo.lense(gal)
        lensedGalEllipse = lensedGal.ellipse()
        lensedGalEllipse.set_facecolor(np.random.rand(3))
        ax.add_artist(lensedGalEllipse)

    plt.show()

    theta = np.linspace(0.0001,100,1000)*u.arcsec
    epsilon = np.array([halo.ellipticity(t, 3*u.Gpc).to_value("") for t in theta])
    gamma = np.array([halo.shear(t, 3*u.Gpc).to_value("") for t in theta])
    kappa = np.array([(halo.surfaceDensity(t) / SIGMA_CRIT(3*u.Gpc, halo.DL)).to_value("") for t in theta])
    mu = 1/((1-kappa)**2 - gamma**2)
    plt.figure(dpi=100)
    plt.plot(theta, epsilon)
    plt.plot(theta, gamma)
    plt.plot(theta, kappa)
    plt.plot(theta, mu)
    plt.legend(["$\epsilon$","$\gamma$","$\kappa$","$\mu$"])
    plt.xlabel("$\\theta$")
    plt.yscale("symlog", linthreshy=1)
    plt.show()

    (halo.rc/halo.DL).to(u.arcsec)
    (4*np.pi*halo.sigmaSquared / c**2 * (3*u.Gpc - halo.DL)/(3*u.Gpc)).to(u.arcsec)
