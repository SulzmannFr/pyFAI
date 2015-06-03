# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Frederic Sulzmann"
__contact__ = "frederic.sulzmann@esrf.fr"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/06/2015"
__status__ = "development"
__docformat__ = 'restructuredtext'

import numpy 
import scipy
import fabio
import pyFAI

from pyFAI.blob_detection import BlobDetection
#from pyFAI.detectors import clac_mask, get_mask
from pyFAI.bilinear import Bilinear
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit
from scipy.ndimage import morphology

from sklearn import cluster, datasets
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale









class Singlemodule(object):
    """
    This class handles a regular grid in front of a detector to calibrate the 
    geometrical distortion of the detector 
    """
    def __init__(self, detector, module):
        """
        @param detector: instance of Detector or its name
        @param module: 
        """
        if isinstance(detector, detectors.Detector):
            self.detector = detector
        else:
            self.detector = detectors.detector_factory(detector)
            
            
        self.module = module
            
        self.module_size = self.detector.MODULE_SIZE
        self.module_gap = self.detector.MODULE_GAP
        self.max_shape = self.detector.MAX_SHAPE
        self.amount = []
        self.center_of_module = []
        self.amount_x = []
        self.amount_y = []
        self.module_area = []
        self.mask = []
        
        
        
    def define_center(self):
        """
        Gives the center of every Module, in an 2d array with the length of 
        the number of modules.
        """
        single_size = numpy.add(self.module_size,self.module_gap)
        total_size = numpy.add(self.max_shape,self.module_gap)
        amount = total_size/single_size
        if amount[0]%2 != 0 and amount[0]%2 != 1:
            print "Error: Module Size, Module Gap, Detector Size does not fit to togeteher"
        elif amount[1]%2 != 0 and amount[1]%2 != 1:
            print "Error: Module Size, Module Gap, Detector Size does not fit to togeteher"
        self.amount =  amount[1]*amount[0]
        self.amount_x = amount[1]
        self.amount_y = amount[0]
        center_of_modules = numpy.zeros([2*self.amount]).reshape(self.amount,2)
        for i in range(self.amount_x):
            for j in range(self.amount_y):
                center_of_modules[j*3+i,1] = ((1+i)*self.module_size[1] + i*self.module_size[1])/2 + i*self.module_gap[1]
                center_of_modules[j*3+i,0] = ((1+j)*self.module_size[0] + j*self.module_size[0])/2 + j*self.module_gap[0]
        self.center_of_module = center_of_modules[self.module]
        
        
    def define_area(self):
        """
        Gives a numpy area with four colums.
        The first colum gives Ymin, the second Ymax, the third Xmin
        and the fourth Xmax of each module. (Ymin,Ymax,Xmin,Xmax)
        """
        if self.amount is None:
            self.define_center()
        X = self.module_size[1]/2.
        Y = self.module_size[0]/2.
        area = numpy.zeros(4*self.amount).reshape(self.amount,4)
        for i in range(self.amount_x):
            for j in range(self.amount_y):
                area[j*3+i,0]= self.center_of_modules[j*3+i,0]-Y
                area[j*3+i,1]= self.center_of_modules[j*3+i,0]+Y
                area[j*3+i,2]= self.center_of_modules[j*3+i,1]-X
                area[j*3+i,3]= self.center_of_modules[j*3+i,1]+X
        self.module_area = area[self.module]
        return area
        
    def single_module(self):
        """
        Gives out a dictionary:
        The name of the keys are "Module_x" with x being an intenger,
        up to the amount of existing Modules.
        The first array gives the center of mass (y,x)
        The second array defines the area (ymin,ymax,xmin,xmax)
        """
        if self.module_area is None:
            self.define_area()
        self.single_module = {'Module_%d'%x:(self.center_of_modules[x],
                              self.module_area[x,:]) 
                              for x in range(self.amount)}
        

    def check_area(self,x,y):
        """
        Checks to which Module a point with (x,y) coordinates contains
        to. Gives 'None' if the point is out of range or lies inside 
        the module gaps.
        """
        if self.module_area is None:
            self.define_area()
        a = area
        for i in range(self.amount):
            if x < self.max_shape[1] and y < self.max_shape[0]:
                if a[i,2] <= x <= a[i,3] and a[i,0] <= y <= a[i,1]:
                    print ('Module_%d') %i
                    return i
                    break
                elif:
                    print ('Point is between the modules')
            elif  x > self.max_shape[1] or y > self.max_shape[0]:
                    print ('Error: Point out of Detector')
        
    def mask_module(self):
        """
        Creates a mask for the used image, which takes only the area 
        of the selected module into account. 
        """
        if self.module_area is None:
            self.define_area()
        x, y = self.max_shape[1],self.maxshape[0] 
        mask = numpy.ones(x,y).reshape(x,y)
        for i in range(area[self.module,2],area[self.module,3]+1):
            for l in range(area[self.module,0],area[self.module,4]+1):
                mask[i,l] = 0
        self.mask = mask
    
    def sieve_points(self,points):
        """
        Sieves out points with (y,x) coordinates if they are not within 
        the module.
        """
        if self.mask is None:
            self.mask_module()
        bilinear = Bilinear(self.mask)
        return [i for i in points if bilinear.f_cy(i)==0]
        
        

