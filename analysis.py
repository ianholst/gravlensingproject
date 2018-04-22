import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmology
import astropy.units as u
from lensing import NFWHalo, IsothermalHalo
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic


# data generation
halo_nfw = NFWHalo(
    M200=1e15*u.solMass,
    C=10,
    DL=1*u.Gpc)

from lensing import BackgroundGalaxy
v=100
backgroundGalaxies = [BackgroundGalaxy(
    Bx=(v*np.random.rand()-v/2)*u.arcsec,
    By=(v*np.random.rand()-v/2)*u.arcsec,
    e1=0,
    e2=0,
    a=1,
    DS=3*u.Gpc) for i in range(1000)]

lensedBackgroundGalaxies = []
for gal in backgroundGalaxies:
    lensedBackgroundGalaxies.append(halo_nfw.lense(gal))

theta_x = np.array([lgal.Tx.to_value(u.arcsec) for lgal in lensedBackgroundGalaxies]) * u.arcsec
theta_y = np.array([lgal.Ty.to_value(u.arcsec) for lgal in lensedBackgroundGalaxies]) * u.arcsec
e1 = np.array([lgal.e1.to_value("") for lgal in lensedBackgroundGalaxies])
e2 = np.array([lgal.e2.to_value("") for lgal in lensedBackgroundGalaxies])
phi = np.array([lgal.phi.to_value(u.rad) for lgal in lensedBackgroundGalaxies])

# Assume we have a dataset of N galaxies with positions (theta_x, theta_y) and ellipticities (e1, e2)
# Assumer they all have same DS and DL is known
DS = 3*u.Gpc
DL = 1*u.Gpc
# Assume these are numpy arrays of length N

theta = np.sqrt(theta_x**2 + theta_y**2)
epsilon = -e1*np.cos(2*phi) - e2*np.sin(2*phi)

# Bin into annuli and calculate mean and standard deviation
theta_bin_edges = np.linspace(1, 70, 30)
epsilon_mean, bin_edges, binnumber = binned_statistic(theta, epsilon, statistic="mean", bins=theta_bin_edges)
epsilon_sigma, bin_edges, binnumber = binned_statistic(theta, epsilon, statistic=np.std, bins=theta_bin_edges)
theta_bin_centers = theta_bin_edges[:-1] + (theta_bin_edges[1] - theta_bin_edges[0])/2


# FITTING
def nfw_ellipticity(theta, M200, C):
    halo = NFWHalo(M200*u.solMass, C, DL)
    return np.array([halo.ellipticity(t*u.arcsec, DS) for t in theta])

def iso_ellipticity(theta, M200, rc):
    halo = IsothermalHalo(M200*u.solMass, rc*u.kpc, DL)
    return np.array([halo.ellipticity(t*u.arcsec, DS) for t in theta])

BOUNDS = [[0,0], [np.inf, np.inf]]
optimalNFWParams, covariance = curve_fit(nfw_ellipticity, theta_bin_centers, epsilon_mean, p0=[1e14, 10], bounds=BOUNDS, sigma=epsilon_sigma)
optimalIsoParams, covariance = curve_fit(iso_ellipticity, theta_bin_centers, epsilon_mean, p0=[1e14, 10], bounds=BOUNDS, sigma=epsilon_sigma)

plt.plot(theta_bin_centers, epsilon_mean)
plt.plot(theta_bin_centers, nfw_ellipticity(theta_bin_centers, 1e15,10))
plt.plot(theta_bin_centers, nfw_ellipticity(theta_bin_centers, *optimalNFWParams))
plt.plot(theta_bin_centers, iso_ellipticity(theta_bin_centers, *optimalIsoParams))
