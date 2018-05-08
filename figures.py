import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmology
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from lensing import *
np.random.seed(1)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 150


# An example of lensing very close to the halo
galaxyNumberDensity = 1000/(100*u.arcsec)**2
viewSize = 100*u.arcsec
Ngal = round((viewSize**2 * galaxyNumberDensity).to_value(""))
zS = 1.0
zL = 0.3
DS = cosmology.Planck15.angular_diameter_distance(zS)
DL = cosmology.Planck15.angular_diameter_distance(zL)
DSL = cosmology.Planck15.angular_diameter_distance_z1z2(zL, zS)


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
    Bx=(viewSize*np.random.rand()-viewSize/2),
    By=(viewSize*np.random.rand()-viewSize/2),
    e1=np.random.normal(0, 0.2),
    e2=np.random.normal(0, 0.2),
    DS=DS) for i in range(Ngal)]

isolensedBackgroundGalaxies = [halo_iso.lense(gal) for gal in backgroundGalaxies]
nfwlensedBackgroundGalaxies = [halo_nfw.lense(gal) for gal in backgroundGalaxies]


def lensingImage(lensedBackgroundGalaxies, v, title):
    # Produce a mock lensing image with a halo and a population of background galaxies
    # v is the view width and a is the size of galaxies

    theta_x = np.array([lgal.Tx.to_value(u.arcsec) for lgal in lensedBackgroundGalaxies]) * u.arcsec
    theta_y = np.array([lgal.Ty.to_value(u.arcsec) for lgal in lensedBackgroundGalaxies]) * u.arcsec
    e1 = np.array([lgal.e1.to_value("") for lgal in lensedBackgroundGalaxies])
    e2 = np.array([lgal.e2.to_value("") for lgal in lensedBackgroundGalaxies])

    # phi = np.array([lgal.phi.to_value(u.rad) for lgal in lensedBackgroundGalaxies])
    # phi = np.arctan(theta_y/theta_x)
    phi = np.arctan2(e2,e1)/2 + np.pi/2
    epsilon = -e1*np.cos(2*phi) - e2*np.sin(2*phi)

    fig = plt.figure()
    plt.gca().set_aspect("equal")
    plt.xlim(-v.to_value(u.arcsec)/2,v.to_value(u.arcsec)/2)
    plt.ylim(-v.to_value(u.arcsec)/2,v.to_value(u.arcsec)/2)
    plt.quiver(theta_x, theta_y, -epsilon*np.sin(phi), epsilon*np.cos(phi),
        pivot="mid", headwidth=0, width=.001)
    plt.xlabel("$\\theta_x$ ($^{\prime\prime}$)")
    plt.ylabel("$\\theta_y$ ($^{\prime\prime}$)")
    plt.title(title)
    plt.savefig("figures/" + title.lower() + "ellipticities.pdf", bbox_inches="tight")
    plt.show()


lensingImage(isolensedBackgroundGalaxies, viewSize, "Isothermal")
lensingImage(nfwlensedBackgroundGalaxies, viewSize, "NFW")
