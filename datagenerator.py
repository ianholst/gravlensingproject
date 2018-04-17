"""
Created by Doyee Byun
April 10, 2018

Goals:
Figure out how to create data using code from lensing.py
define initial conditions for creating data
galaxy locations, lens profile, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.cosmology as cosmology
from astropy.visualization import astropy_mpl_style
import astropy.units as u
import astropy.constants as const

#import profiles from lensing.py
from lensing import NFWHalo
from lensing import IsothermalHalo
from lensing import BackgroundGalaxy

# data format is theta_x, theta_y, e1, e2

#test parameters for lens
M200_0 = 100*u.solMass
c_param = 0.5
dl = 1*u.Gpc

datafile = open('output1.txt','w')
nfw_1 = NFWHalo(M200_0,c_param,dl)
backgroundGalaxies = [BackgroundGalaxy(
    beta_x=(100*np.random.rand()-50)*u.arcsec,
    beta_y=(100*np.random.rand()-50)*u.arcsec,
    e1=0,
    e2=0,
    a=1*u.arcsec,
    DS=10*u.Gpc) for i in range(300)]
for i in range(0,len(backgroundGalaxies)):
    newGalx = nfw_1.lense(backgroundGalaxies[i])
    datafile.write(newGalx.theta_x,"\t",newGalx.theta_y,"\t",newGalx.e1,"\t",newGalx.e2,"\t\n")

#Isothermal Profile lensing will be added once the paramaters are modified
#to be identical to those of NFW

datafile.close()
