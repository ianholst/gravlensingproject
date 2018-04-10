import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.cosmology as cosmology
from astropy.visualization import astropy_mpl_style
import astropy.units as u
import astropy.constants as const

from lensing import NFWHalo
from lensing import IsothermalHalo

# data format is theta_x, theta_y, e1, e2
datafile = open('output.txt','w')
datafile.write("Hello world")
datafile.close()
