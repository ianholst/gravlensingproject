import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmology
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from lensing import *

# here is an example of lensing very close to the halo
galaxyNumberDensity = 50/u.arcmin**2
viewSize = 1000*u.arcsec
Ngal = round((viewSize**2 * galaxyNumberDensity).to_value(""))
zS = 1.0
zL = 0.3
DS = cosmology.Planck15.comoving_distance(zS) / (1 + zS)
DL = cosmology.Planck15.comoving_distance(zL) / (1 + zL)

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
