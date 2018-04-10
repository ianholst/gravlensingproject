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

# data format is theta_x, theta_y, e1, e2
datafile = open('output.txt','w')
datafile.write("Hello world") #Dummy code for testing write function
datafile.close()
