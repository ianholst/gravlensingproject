import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmology
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from lensing import *
np.random.seed(1)

### DATA GENERATION ###
galaxyNumberDensity = 50/u.arcmin**2
viewSize = 1000*u.arcsec
Ngal = round((viewSize**2 * galaxyNumberDensity).to_value(""))
zS = 1.0
zL = 0.3
DS = cosmology.Planck15.comoving_distance(zS) / (1 + zS)
DL = cosmology.Planck15.comoving_distance(zL) / (1 + zL)

lensHalo = NFWHalo(
    M200=1e15*u.solMass,
    C=10,
    DL=DL)

backgroundGalaxies = [BackgroundGalaxy(
    Bx=(viewSize*np.random.rand() - viewSize/2),
    By=(viewSize*np.random.rand() - viewSize/2),
    e1=np.random.normal(0, 0.2),
    e2=np.random.normal(0, 0.2),
    DS=DS) for i in range(Ngal)]

lensedBackgroundGalaxies = [lensHalo.lense(gal) for gal in backgroundGalaxies]

theta_x = np.array([lgal.Tx.to_value(u.arcsec) for lgal in lensedBackgroundGalaxies]) * u.arcsec
theta_y = np.array([lgal.Ty.to_value(u.arcsec) for lgal in lensedBackgroundGalaxies]) * u.arcsec
e1 = np.array([lgal.e1.to_value("") for lgal in lensedBackgroundGalaxies])
e2 = np.array([lgal.e2.to_value("") for lgal in lensedBackgroundGalaxies])
phi = np.array([lgal.phi.to_value(u.rad) for lgal in lensedBackgroundGalaxies])


### ANALYSIS ###

# Assume we have a dataset of N galaxies with positions (theta_x, theta_y) and ellipticities (e1, e2)
# Assume they all have same known DS and DL
# Assume these are numpy arrays of length N

# Calculate radial position theta and magnitude of ellipticity epsilon
theta = np.sqrt(theta_x**2 + theta_y**2)
epsilon = -e1*np.cos(2*phi) - e2*np.sin(2*phi)
# Create theta bins
bin_start = 10*u.arcsec
bin_stop = viewSize/np.sqrt(2)
N_bins = 40
theta_bin_edges = np.linspace(bin_start, bin_stop, N_bins)
# Bin into annuli and calculate mean and standard deviation
epsilon_mean, bin_edges, binnumber = binned_statistic(theta, epsilon, statistic="mean", bins=theta_bin_edges)
epsilon_sigma, bin_edges, binnumber = binned_statistic(theta, epsilon, statistic=np.std, bins=theta_bin_edges)
# Calculate number of galaxies in each bin
N_in_bin, bin_edges = np.histogram(theta, bin_edges)
# Calculate centers of bins
theta_bin_centers = theta_bin_edges[:-1] + (theta_bin_edges[1] - theta_bin_edges[0])/2
# Save out bin centers, ellipticities, and uncertainties for analysis in Mathematica
np.savetxt("data.csv", np.stack([theta_bin_centers, epsilon_mean, epsilon_sigma/np.sqrt(N_in_bin)], 1), delimiter=", ")


### FITTING ###
# python hates fitting

# def nfw_ellipticity(theta, M200, C):
#     halo = NFWHalo(M200*u.solMass, C, DL)
#     return np.array([halo.ellipticity(t*u.arcsec, DS) for t in theta])
#
# def iso_ellipticity(theta, M200, rc):
#     halo = IsothermalHalo(M200*u.solMass, rc*u.kpc, DL)
#     return np.array([halo.ellipticity(t*u.arcsec, DS) for t in theta])
#
# BOUNDS = [[0,0], [np.inf, np.inf]]
# optimalNFWParams, covariance = curve_fit(nfw_ellipticity, theta_bin_centers, epsilon_mean, p0=[1e16, 20], bounds=BOUNDS, sigma=epsilon_sigma)
# optimalIsoParams, covariance = curve_fit(iso_ellipticity, theta_bin_centers, epsilon_mean, p0=[1e16, 20], bounds=BOUNDS, sigma=epsilon_sigma)
#
# plt.figure()
# plt.plot(theta_bin_centers, nfw_ellipticity(theta_bin_centers, 1e15, 10))
# plt.plot(theta_bin_centers, epsilon_mean)
# plt.plot(theta_bin_centers, nfw_ellipticity(theta_bin_centers, *optimalNFWParams))
# plt.plot(theta_bin_centers, iso_ellipticity(theta_bin_centers, *optimalIsoParams))
# # plt.scatter(theta, epsilon, s=1)
# plt.xlabel("$\\theta$")
# plt.ylabel("$\epsilon$")
# plt.legend(["original", "binned data", "nfw fit", "isothermal fit", "lensed galaxies"])
# plt.show()
