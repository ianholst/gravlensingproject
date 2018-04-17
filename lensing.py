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


def SIGMA_CRIT(DS, DL):
    DSL = DS - DL
    return c**2 / (4 * np.pi * G) * (DS / (DSL * DL))


class NFWHalo:

    def __init__(self, M200, c_param, DL):
        self.M200 = M200
        self.c = c_param     #changed variable name for clarification
        self.delta_c = (200/3) * c_param**3 / (np.log(1+c_param) - c_param/(1+c_param))
        self.rs = 1/c_param * ((3 * M200) / (800 * np.pi * RHO_CRIT))**(1/3)
        self.DL = DL


class IsothermalHalo:

    def __init__(self, sigma, rc, DL):
        self.sigma = sigma
        self.rc = rc
        self.DL = DL

    def surfaceDensity(self, R):
        return self.sigma**2 / (2 * G * np.sqrt(R**2 + self.rc**2))

    def averageSurfaceDensity(self, R):
        return self.sigma**2 * (np.sqrt(R**2 + self.rc**2) - self.rc) / (G * R**2)

    def shear(self, R, DS):
        return (self.averageSurfaceDensity(R) - self.surfaceDensity(R)) / SIGMA_CRIT(DS, self.DL)

    def ellipticity(self, R, DS):
        reducedShear = self.shear(R, DS) / (1 - self.averageSurfaceDensity(R) / SIGMA_CRIT(DS, self.DL))
        return 2 * reducedShear / (1 + reducedShear**2)

    def deflection(beta, ):
        dd

    def lense(self, galaxy):
        beta = np.sqrt(galaxy.beta_x**2 + galaxy.beta_y**2)
        phi = np.degrees(np.arctan2(galaxy.beta_y, galaxy.beta_x))
        epsilon = self.ellipticity(self.DL*beta.to_value(u.rad), galaxy.DS)
        area_scaling = np.sqrt(1 / (1 - epsilon))
        # deflection angle: beta -> theta
        return LensedBackgroundGalaxy(galaxy.beta_x, galaxy.beta_y, galaxy.a, galaxy.e1, galaxy.e2, galaxy.DS)


class BackgroundGalaxy:

    def __init__(self, beta_x, beta_y, a, e1, e2, DS):
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.a = a
        self.e1 = e1
        self.e2 = e2
        self.DS = DS

    def ellipse(self):
        a = 1
        b = 1
        phi = 1
        x = (self.DS * self.beta_x.to_value(u.rad)).to_value(u.Mpc)
        y = (self.DS * self.beta_y.to_value(u.rad)).to_value(u.Mpc)
        return Ellipse(xy=[x,y], width=a, height=b, angle=phi)

class LensedBackgroundGalaxy:

    def __init__(self, theta_x, theta_y, a, e1, e2, DS):
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.a = a
        self.e1 = e1
        self.e2 = e2
        self.DS = DS

    def ellipse(self):
        a = .01
        b = .01
        phi = 1
        x = (self.DS * self.theta_x.to_value(u.rad)).to_value(u.Mpc)
        y = (self.DS * self.theta_y.to_value(u.rad)).to_value(u.Mpc)
        return Ellipse(xy=[x,y], width=a, height=b, angle=phi)


halo = IsothermalHalo(sigma=1*u.m/u.s, rc=1*u.Mpc, DL=1*u.Gpc)
backgroundGalaxies = [BackgroundGalaxy(
    beta_x=(100*np.random.rand()-50)*u.arcsec,
    beta_y=(100*np.random.rand()-50)*u.arcsec,
    e1=0,
    e2=0,
    a=1*u.arcsec,
    DS=10*u.Gpc) for i in range(300)]

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, aspect="equal")

for gal in backgroundGalaxies:
    lensedGal = halo.lense(gal)
    lensedGalEllipse = lensedGal.ellipse()
    lensedGalEllipse.set_facecolor(np.random.rand(3))
    ax.add_artist(lensedGalEllipse)

plt.show()
