import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmology
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from lensing import *
np.random.seed(0)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 150


# here is an example of lensing very close to the halo
galaxyNumberDensity = 1000/(100*u.arcsec)**2
viewSize = 100*u.arcsec
Ngal = round((viewSize**2 * galaxyNumberDensity).to_value(""))
zS = 1.0
zL = 0.3
DS = cosmology.Planck15.comoving_distance(zS) / (1 + zS)
DL = cosmology.Planck15.comoving_distance(zL) / (1 + zL)

halo_iso = IsothermalHalo(
    M200=1e15*u.solMass,
    rc=10*u.kpc,
    DL=DL)

halo_nfw = NFWHalo(
    M200=1e15*u.solMass,
    C=10,
    DL=DL)

def plotProperties(halo, start, stop, step, DS, name):
    # Plot the shear, ellipticity, convergence, and magnification for the halo
    theta = np.linspace(start, stop, step)
    epsilon = np.array([halo.ellipticity(t, DS) for t in theta])
    gamma = np.array([halo.shear(t, DS) for t in theta])
    kappa = np.array([(halo.surfaceDensity(t) / SIGMA_CRIT(DS, halo.DL)).to_value("") for t in theta])
    mu = 1/((1-kappa)**2 - gamma**2)
    plt.figure()
    plt.plot(theta, epsilon)
    plt.plot(theta, gamma)
    plt.plot(theta, kappa)
    # plt.plot(theta, mu)
    plt.legend(["$\epsilon$", "$\gamma$", "$\kappa$", "$\mu$"])
    plt.xlabel("$\\theta$ ($^{\prime\prime}$)")
    plt.ylim(-1,3)
    plt.xlim(start.to_value(), stop.to_value())
    plt.savefig("figures/" + name + ".pdf", bbox_inches="tight")
    return plt.show()

plotProperties(halo_iso, 0.01*u.arcsec, viewSize, 500, DS, "isothermalproperties")
print("theta_c:", halo_iso.Tc.to(u.arcsec))
print("theta_0:", halo_iso.T0(DS).to(u.arcsec))

plotProperties(halo_nfw, 0.01*u.arcsec, viewSize, 500, DS, "nfwproperties")
print("theta_s:", halo_nfw.Ts.to(u.arcsec))


backgroundGalaxies = [BackgroundGalaxy(
    Bx=(viewSize*np.random.rand()-viewSize/2)*u.arcsec,
    By=(viewSize*np.random.rand()-viewSize/2)*u.arcsec,
    e1=0,
    e2=0,
    DS=DS) for i in range(Ngal)]

def lensingImage(halo, backgroundGalaxies, v, a=1):
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


lensingImage(halo_iso, backgroundGalaxies, viewSize)
lensingImage(halo_nfw, backgroundGalaxies, viewSize)
