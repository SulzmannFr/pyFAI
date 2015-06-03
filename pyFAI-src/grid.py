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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/05/2015"
__status__ = "development"
__docformat__ = 'restructuredtext'

import logging
import numpy
import os
import time

from . import detectors
from . import units
from .third_party import six
StringTypes = (six.binary_type, six.text_type)

logger = logging.getLogger("pyFAI.grid")

class Grid(object):
    """
    This class handles a regular grid in front of a detector to calibrate the 
    geometrical distortion of the detector 
    """
    def __init__(self, detector, image, mask=None, pitch=None, invert=False):
        """
        @param detector: instance of Detector or its name
        @parma image: 2d array representing the image  
        @param mask:
        @param pitch: 2-tuple representing the grid spacing in (y, x) coordinates, in meter
        @param invert: set to true if the image of the grid has regular dark spots (instead of bright points) 
        """
        if isinstance(detector, detectors.Detector):
            self.detector = detector
        else:
            self.detector = detectors.detector_factory(detector)

        if isinstance(image, numpy.ndarray):
            self.image = image
        else:
            self.image = fabio.open(image).data

        if mask is not None:
            if isinstance(mask, numpy.ndarray):
                self.mask = mask
            else:
                self.mask = fabio.open(mask).data.astype(bool)
            if self.detector.mask is not None:
                self.mask = numpy.logical_or(self.detector.mask, self.mask)
        else:
            self.mask = numpy.zeros_like(self.image, bool)
        if invert:
            self.image = self.image.max() - self.image
        self.pitch = tuple(pitch[0], pitch[-1])
        self.coordinates=[]
        self.distances=[]
        self.indices=[]
        self.mean_distance=[]
        self.mean_distance_min=[]
        self.mean_distance_max=[]
        self.good_peaks=[]
        self.mean_vec_y = []
        self.mean_vec_x = []
        self.center_of_grid = []
        self.grid = {}
        
            
    def aply_threshold(self, level=None, percentile=None):
        """
        Segment the image with a single threshold
        @param 
        @param 
        """
        if percentile and not level:
            data = self.image[self.mask].flatten()
            data.sort()
            level = data[int(len(data) * percentile / 100.)]
        thres = level
        mask = img>thres
        lbl,n = scipy.ndimage.label(mask)
        b = numpy.array(scipy.ndimage.measurements.center_of_mass(img, lbl, range(n)))
        v=np.std(Image)
        self.coordinates = numpy.array(b)

    def nearest_neighbors(self):
        """
        Defines the nearest neighbors and exculde not suitible,
        under condition of historamm
        """
        nbrs = NearestNeighbors(n_neighbors = 5, algorithm='ball_tree').fit(self.coordinates)
        self.distances, self.indices = nbrs.kneighbors(self.coordinates)
        hist, bin_edges = np.histogram(self.distances[:,1:5], bins=1000)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2));
        c=np.zeros(2*len(self.coordinates)).reshape(len(self.coordinates),2)
        p0 = [hist[1:].max(), bin_centres[hist[1:].argmax()+1], 1.]
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        test = 4
        self.mean_distance = coeff[1]
        self.mean_distance_min = coeff[1]-abs(test*coeff[2])
        self.mean_distance_max = coeff[1]+min(test*abs(coeff[2]),coeff[1]/np.sqrt(2))
        for i in range(len(self.coordinates)):
            for l in range(0,5):
                if self.distance_min <= self.distance[i,l] <= self.distances_max:
                    c[i]=c1[i]
        self.good_peaks = c[c.all(1)]
   
    def mean_of_vectors(self):
        """
        Gives the averaged vector of the grid in x and y axis. 
        They are called mean_vec_x, mean_vec_y, respectively
        """ 
        maax = self.mean_distance_max
        miin = self.mean_distance_min
        all_vectors = np.zeros(8*len(self.coordinates)).reshape(4*len(self.coordinates),2)
        for l in range(1,5):
            for j in range(len(self.coordinates)):
                all_vectors[(l-1)*len(self.coordinates)+j,:] = self.coordinates[j,:]-self.coordinates[self.indices[j,l],:]
        for k in range(len(all_vectors)):
            if abs(all_vectors[k,1]) > maax or 3 < abs(all_vectors[k,1]) < miin:
                all_vectors[k,:] = 0 
            elif abs(all_vectors[k,0]) > maax or 3 < abs(all_vectors[k,0]) < miin:
                all_vectors[k,:] = 0 
        all_vectors = all_vectors[all_vectors.all(1)]
        ms = KMeans(n_clusters=8).fit(all_vectors)
        Mean =  ms.cluster_centers_
        self.mean_vec_y = np.array([0,0])
        self.mean_vec_x = np.array([0,0])
        for i in range(0,8):
            for j in range(1,8):
                if abs(Mean[i,0])-abs(Mean[j,0])<1 and miin <abs(Mean[i,0])< maax and miin <abs(Mean[j,0])< maax and abs(Mean[i,1])<1 and abs(Mean[j,1])<1:
                    self.mean_vec_y= np.add(abs(Mean[i]),abs(Mean[j]))/2
                if abs(Mean[i,1])-abs(Mean[j,1])<1 and miin < abs(Mean[i,1])< maax and miin < abs(Mean[j,1])< maax and abs(Mean[i,0])<1 and abs(Mean[j,0])<1:
                    self.mean_vec_x= np.add(abs(Mean[i]),abs(Mean[j]))/2
    #Achtung nur absolut Werte!!!!!

    def define_center_of_grid(self):
        """
        Finds the coordinate of the peak, which is closest to the center
        of the Module
        """
        f = numpy.append(self.good_peaks, self.Singlemodule.center_of_modules,axis=0)
        number = self.Singlemodule.amount 
        nbrs = NearestNeighbors(n_neighbors = 2, algorithm='ball_tree').fit(f)
        distances, indices = nbrs.kneighbors(f)
        Center = numpy.zeros(2*number).reshape(number,2)
        for i in range(number):
            Center[i] = self.good_peaks[indices[len(indices)-(number+1)+i,1]]
        self.center_of_grid = Center
        #Achtung Center of modules von nicht genutzten Modulen muessen noch eliminiert werden
    
    def spacing_grid(self):
        """
        Constructs the dictonary where one will save the the points for 
        the grid. There are as many 'keys' as modules. For example they 
        are named Module_1 and it's 'values' will be a numpy array with 
        the (y,x) coordinates of the points of the grid
        """
        for i in range(self.Singlemodule.amount):
            self.grid['Module_%d' %i] = 0
            
            
    def build_rough_grid(self,module):
        """
        Build the grid for in Order to the given Module Size and Center
        of the Grid. Important these are not identical with the center 
        the Modules!
        @Parameter module: Simple Intenger number
        """
        if self.grid is None:
            self.spacing_grid()
        X = int(MODULE_SIZE[1]/int(2*self.mean_vec_x[1]))+1
        Y = int(MODULE_SIZE[0]/int(2*self.mean_vec_y[0]))+!
        Grid = numpy.zeros(2*(4*X*Y)).reshape(4*X*Y,2)
        for i in range(-Y,Y): 
            for j in range(-X,X): 
                Grid[(Y+i)*2*X+X+j] = i*self.mean_vec_y+j*self.mean_vec_x+self.center_of_grid[module]
        self.grid['Module_%d' %module] = Grid 


    def find_missing_points(self):
        """
        """
        for i in range(self.Singlemodule.amount):
            if self.grid['Module_%d' %i] != 0:
                ensemble = np.append(self.coordinates,self.grid['Module_%d' %module],axis=0)
                nbrs = NearestNeighbors(n_neighbors = 2, algorithm='ball_tree').fit(ensemble)
                distances, indices = nbrs.kneighbors(ensemble)
                perfect_peak=np.zeros(2*len(self.coordinates)).reshape(len(self.coordinates),2)
                Ende=np.zeros(2*len(self.coordinates)).reshape(len(self.coordinates),2)
                for i in range(len(self.coordinates)):
                    if indices[i,1]>len(self.coordinates):
                        perfect_peak[i,:]=Besser[i,:]
                        Ende[i,:]=Besser[indices[i,1],:]
                        Ende=Ende[Ende.all(1)]
                        perfect_peaks = perfect_peak[perfect_peak.all(1)]

    
    
    
    
    def alginement(self):
        """
        """
        self.threshold()
        self.nearest_neighbors()
        self.mean_of_vectors()
        raise NotImplemented("TODO")
