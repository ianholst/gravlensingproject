import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

c = 1
G = 1
RHO_CRIT = 1

def SIGMA_CRIT(DS, DL):
    DSL = DS - DL
    return c**2 / (4 * np.pi * G) * (DS / (DSL * DL))


class NFWHalo:

    def __init__(self, M200, c, DL):
        self.M200 = M200
        self.c = c
        self.delta_c = (200/3) * c**3 / (np.log(1+c) - c/(1+c))
        self.rs = 1/c * ((3 * M200) / (800 * np.pi * RHO_CRIT))**(1/3)
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

    def lense(self, galaxy):
        R = np.sqrt(galaxy.x**2 + galaxy.y**2)
        phi = np.degrees(np.arctan2(galaxy.y, galaxy.x))
        epsilon = self.ellipticity(R, galaxy.DS)
        area_scaling = np.sqrt(1 / (1 - epsilon))
        return Ellipse(xy=[galaxy.x, galaxy.y], width=(1-epsilon)*galaxy.size, height=galaxy.size, angle=phi)


class BackgroundGalaxy:

    def __init__(self, e_intrinsic, size, x, y, DS):
        self.DS = DS
        self.e_intrinsic = e_intrinsic
        self.size = size
        self.x = x
        self.y = y
        self.ellipse = Ellipse(xy=[x,y], width=size, height=size, angle=0)



halo = IsothermalHalo(1, 1, 1)
backgroundGalaxies = [BackgroundGalaxy(1, 3, 100*np.random.rand()-50, 100*np.random.rand()-50, 10) for i in range(300)]

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, aspect='equal')

for gal in backgroundGalaxies:
    galellipse = halo.lense(gal)
    galellipse.set_facecolor(np.random.rand(3))
    ax.add_artist(galellipse)

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

plt.show()
