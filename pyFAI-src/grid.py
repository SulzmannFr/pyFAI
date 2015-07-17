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


import pyFAI
import numpy 
import scipy
import fabio


from pyFAI.blob_detection import BlobDetection
from pyFAI import detectors
from pyFAI.bilinear import Bilinear
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import asarray as ar,exp
from scipy.optimize import curve_fit, fmin
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


import logging
import numpy
import os
import time

import numpy.ma as m

#from . import detectors
#from . import units
#from .third_party import six
#StringTypes = (six.binary_type, six.text_type)

#logger = logging.getLogger("pyFAI.grid")

class Grid(object):
    """
    This class handles a regular grid in front of a detector to calibrate the 
    geometrical distortion of the detector 
    """
    def __init__(self, detector, image, pitch, beamcenter, detectordistance, mask=None, invert=False): 
        """
        @param detector: instance of Detector or its name
        @param image: 2d array representing the image  
        @param mask:
        @param pitch: floating number in millimeter, representing distance between holes
        @param beamcenter: 2d array for the position of the beam, (y,x)-coordinate in Px
        @param detectordistance: floating number in meter representing distance between detector and scatterin object
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
        
        
        self.pitch = pitch
        self.detectordistance = detectordistance
        self.beamcenter = beamcenter
        self.coordinates=[]
        self.good_peaks=[]
        self.mean_vec = []
        self.module_translation = []
        self.grid_angle =[]
        self.grid =[]
        
   
            
    def apply_threshold(self, level=None, percentile=None, module=None):
        """
        Segment the image with a single threshold
        @param  level integernumber for the intensity, should be high 
                enough to get rid of the diffused points
        @param  percentile number between 0 and 100, can be easily 
                calculated with the proportion between hole and pitch.
        """
        if percentile and not level:
            data = self.image[self.mask].flatten()
            data.sort()
            level = data[int(len(data) * percentile / 100.)]
        if module is not None:    
            m = Singlemodule(self.detector,module).mask_module()
            for i in range(numpy.shape(self.image)[1]):
                for j in range(numpy.shape(self.image)[0]):
                    if m[j,i] == 1:
                        self.image[j,i] = 0
        thres = level
        mask = self.image>thres
        lbl,n = scipy.ndimage.label(mask)
        b = numpy.array(scipy.ndimage.measurements.center_of_mass(self.image, lbl, range(n)))
        self.coordinates = numpy.array(b)

    def define_nearest_neighbors(self):
        """
        Defines the nearest neighbors and exculde not suitible,
        under condition of historamm
        """
        if self.coordinates == []:
            self.apply_threshold()
        distances, indices = Grid.get_nearest_neighbors(self.coordinates,5)
        hist, bin_edges = numpy.histogram(distances[:,0:4], bins=1000)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
        def gauss(x, *p):
            A, mu, sigma = p
            return A*numpy.exp(-(x-mu)**2/(2.*sigma**2));
        c=self.coordinates
        p0 = [hist[1:].max(), bin_centres[hist[1:].argmax()+1], 1.]
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
        mean_distance = coeff[1]
        mean_distance_min = coeff[1]-abs(4*coeff[2])
        mean_distance_max = coeff[1]+4*abs(coeff[2])
        mask2 = numpy.where(numpy.logical_and(distances>=mean_distance_min, distances<=mean_distance_max).sum(axis=-1)==4)[0]
        self.good_peaks = c[mask2,:]
    
    def get_angle(self,x=None):
        """
        """
        if x is None:
             x = self.good_peaks
        distances, indices = Grid.get_nearest_neighbors(x,4)
        a = len(x)
        mean = distances[:,0].mean()
        coordinates = numpy.zeros((a,4,2))
        numbers = numpy.zeros((a,3,2))
        phiy, phix = numpy.zeros(a),numpy.zeros(a)
        coorx, coory = numpy.zeros((a,2)),numpy.zeros((a,2))
        for i in range(a):
            for j in range(3):
                coordinates[i,j+1,:] = x[i,:] - x[indices[i,j],:]
                numbers[i,j,:] =  numpy.around(coordinates[i,j+1]/mean)
                if numbers[i,j,0] == 1 and numbers[i,j,1] == abs(0):
                    phiy[i] = numpy.rad2deg(numpy.arctan(coordinates[i,j+1,1]/coordinates[i,j+1,0]))
                    coory[i,:] = coordinates[i,j+1,:]
                if numbers[i,j,1] == 1 and numbers[i,j,0] == abs(0):
                    phix[i] = numpy.rad2deg(numpy.arctan(coordinates[i,j+1,0]/coordinates[i,j+1,1]))
                    coorx[i,:] = coordinates[i,j+1,:]
        coorx = coorx[coorx != 0]
        coory = coory[coory != 0]
        phix = phix[phix != 0]
        phiy = phiy[phiy != 0]
        coorx = coorx.reshape(len(phix),2)
        coory = coory.reshape(len(phiy),2)
        vec = numpy.array([coory.mean(axis=0),coorx.mean(axis=0)])
        phi = numpy.sign(phix.mean())*(abs(phix.mean())+abs(phiy.mean()))/2
        self.mean_vec = vec
        return phi, vec;
        

    
    def minimize(self):
        """
        taranslation,angle,gap has to be optimised
        x,detectordistance,beamcenter,pitch constant
        gap & detectordistance in m
        """
        if self.good_peaks == []:
            self.define_nearest_neighbors()
        if self.mean_vec == []:
            self.get_angle()
        x = self.good_peaks
        beamcenter= self.beamcenter
        pitch = self.pitch
        detectordistance = self.detectordistance
        # Initial Guesses:
        beamcentery,beamcenterx = self.beamcenter
        gap = Grid.get_grid_detector_gap(x,detectordistance,pitch)
        c = numpy.array([gap, pitch,detectordistance,beamcentery,beamcenterx])
        c = tuple(numpy.append(x.flatten(),c))
        transy, transx = Grid.find_center_of_mass(self.good_peaks)
        phi, vector = self.get_angle(x) 
        p0 = numpy.array([transx,transy,phi])
        print p0
        def grid(v,*c):
            """
            """
            transx,transy,phi = v
            coord,gap,pitch,detectordistance = c[:-5],c[-5],c[-4],c[-3]
            beamcentery, beamcenterx = c[-2],c[-1]
            beamcenter =  numpy.array([beamcentery, beamcenterx])
            x = numpy.asarray(coord).reshape(len(coord)/2,2)
            gap = Grid.get_grid_detector_gap(x,detectordistance,pitch)
            x,transx,transy,bc = Grid.correct_points(x, detectordistance,
                                            beamcenter,pitch, gap=gap,
                                            shiftx=transx, shifty=transy)
            phi, control_vec = self.get_angle(x)
            ind_control = Grid.get_indices(x,transx,transy,control_vec)
            vec = Grid.get_pitch_vector(pitch,phi)
            ind_grid = Grid.get_indices(x,transx,transy,vec)
            delta = ind_control-ind_grid
            err = (delta[0,:]**2 + delta[1,:]**2).sum()
            return err;
        opt = fmin(grid,p0,c)
        print opt
        good_peak, bc = Grid.correct_points(opt[0:2], detectordistance, beamcenter, pitch, gap=gap)
        top = Grid.get_detectorplane_position(opt[0:2],bc,detectordistance,pitch,gap=gap)
        print top
        top = top[0]+top[1]
        vector =   Grid.get_pitch_vector_at_detector(pitch,opt[2],0.647, gap)
        print vector 
        u = Grid.get_indices(self.good_peaks,top[1],top[0],vector)
        print u.min(),u.max()
        print top
        self.grid = numpy.dot(u,vector)+top
        self.module_translation = (self.grid-x).mean(axis=0)
        self.grid_angle = opt[2]
        return self.module_translation,self.grid_angle;
        

        

    
    @staticmethod
    def get_nearest_neighbors(x,neighbors):
        """
        """
        nbrs = NearestNeighbors(n_neighbors = neighbors, algorithm='ball_tree').fit(x)
        dis, ind = nbrs.kneighbors(x)
        distances = dis[:,1:neighbors]
        indices = ind[:,1:neighbors]
        return distances, indices;
        
    @staticmethod
    def get_grid_detector_gap(x,detectordistance,pitch):
        """
        Calculates the grid detector distance, due to the theorem of Thales
        @param x: 2d array of (y,x) coordinates
        @param detectordistance: Distance between detector and sample in m
        @param pitch: Assuming a quadratic gird, pinnholedistance in mm
        @return: grid detector distance in m
        """
        
        distances, indices = Grid.get_nearest_neighbors(x,5)
        mean = distances[:,0].mean()
        distances_ab = numpy.zeros(distances.shape)
        numbers = numpy.zeros((len(x),4,2))
        norm = pitch*10**(3)/172.
        for i in range(len(x)):
            for j in range(4):
                numbers[i,j,:] = numpy.around((x[i]-x[indices[i,j]])/mean)
                distances_ab[i,j] = numpy.sqrt((numbers[i,j,0]*norm)**2+(numbers[i,j,1]*norm)**2)
        verhaltnis = (distances_ab/distances).mean()
        gap = detectordistance-(detectordistance*verhaltnis)
        return gap;
        
    @staticmethod
    def get_pitch_vector(pitch,angle):
        """
        pitch in mm
        """
        phi = numpy.deg2rad(angle)
        norm = pitch*10**(3)/172.
        vector = numpy.array([[norm*numpy.cos(phi),abs(norm*numpy.sin(phi))],
                            [-norm*numpy.sin(phi),norm*numpy.cos(phi)]])
        return vector;
    
    @staticmethod
    def get_pitch_vector_at_detector(pitch,angle,detectordistance,gap):
        """
        pitch in mm
        """
        phi = numpy.deg2rad(angle)
        norm = pitch*10**(3)/172.*(gap/detectordistance+1)
        vector = numpy.array([[norm*numpy.cos(phi),norm*numpy.sin(phi)],
                            [-norm*numpy.sin(phi),norm*numpy.cos(phi)]])
        return vector;

    @staticmethod
    def get_detectorplane_position(x,beamcenter,detectordistance,pitch,gap=None):
        """
        """
        if gap is None:
            gap = Grid.get_grid_detector_gap(x,detectordistance,pitch)
        peaks = x-beamcenter
        peaks = numpy.append([peaks],[beamcenter],axis=0)
        z = numpy.ones((len(peaks),2))
        z[:,0] = -((detectordistance-gap)/172e-6)
        peaks = numpy.append(peaks,z,axis=1)
        d = detectordistance/172e-6
        Projektion = numpy.identity(4)
        Projektion[3,2] = -1/d
        Projektion[3,3] = 0
        c = numpy.dot(Projektion,peaks.T[:]).T
        c = c[:]/c[0,3]
        good_peaks = c[:,0:2]
        return good_peaks;
        
    @staticmethod
    def find_center_of_mass(x):
        """
        Returns that point of the given one, which is closest to the 
        center of mass.
        @param x: 2d array of (y,x) coordinates
        @return: 1d array with (y,x) coordinates
        
        """
        p = cluster.KMeans(n_clusters=1).fit(x).cluster_centers_
        f = numpy.append(x, p,axis=0)
        distances, indices = Grid.get_nearest_neighbors(f,2)
        x0 = x[indices[-1,0]]
        return x0;
        
        

    @staticmethod
    def get_indices(x,xcoordinate,ycoordinate,vector):
        """
        @param x: 2d array of (y,x) coordinates
        @param xcoordinate: shift of the starting point in x-direction
        @param ycoordinate: shift of the starting point in y-direction
        @param vecor: 2x2 matrix with the supporting vectors of the grid
        @return: 2d array with floating points, with gives the position 
                of the points accortind to the supporting vector and 
                starting point
        """
        x0 = numpy.array([ycoordinate,xcoordinate])
        invT = numpy.linalg.inv(vector)
        u = numpy.dot(invT,(x-x0).T).T
        return u;
    
    @staticmethod
    def correct_points(x, detectordistance, beamcenter, pitch, gap=None , shiftx=None,shifty=None):
        """
        """
        if gap is None:
            gap = Grid.get_grid_detector_gap(x,detectordistance,pitch)
        if shiftx is not None and shifty is not None:
            shift = numpy.array([shifty,shiftx])
            x = numpy.append(x,[shift],axis=0)
        if len(x) == 2:
            peaks = numpy.append([x],[beamcenter],axis=0)
        else:
            peaks = numpy.append(x,[beamcenter],axis=0)
        z = numpy.ones((len(peaks),2))
        z[:,0] = -(detectordistance/172e-6)
        peaks = numpy.append(peaks,z,axis=1)
        d = (detectordistance-gap)/172e-6
        Projektion = numpy.identity(4)
        Projektion[3,2] = -1/d
        Projektion[3,3] = 0
        c = numpy.dot(Projektion,peaks.T[:]).T
        c = c[:]/c[0,3]
        if shiftx is not None and shifty is not None:
            shifty,shiftx = c[-2,0:2]
            good_peaks = c[:-2,0:2]
            bc = c[-1,0:2]
            return good_peaks,shiftx,shifty, bc;
        elif len(x) == 2:
            good_peaks = c[0,0:2]
            bc = c[-1,0:2]
            return good_peaks, bc;
        else:
            good_peaks = c[:-2,0:2]
            bc = c[-1,0:2]
            return good_peaks, bc;
    
        
        
        
        
    def visualize_grid(self):
        """
        Plots the the the calculated Values as far as calculated.
        """
        plt.imshow(self.image, aspect='auto')
        plt.show()
        if self.coordinates != []:
            plt.plot(self.coordinates[:,1],self.coordinates[:,0],"ow")
            
        if self.good_peaks != []:
            plt.plot(self.good_peaks[:,1],self.good_peaks[:,0],"or")
        
        if self.grid != []:
            plt.plot(self.grid[:,1],self.grid[:,0],"og")

    
    def alginement(self):
        """
        """
        self.threshold()
        self.nearest_neighbors()
        self.mean_of_vectors()
        raise NotImplemented("TODO")

correction1 = numpy.zeros((9,3))
g = Grid(pyFAI.detectors.Pilatus2M(),'grid1_wax_0004p.cbf',2.54,numpy.array([1030,734]),0.647)
g.apply_threshold(level=10)
correction1 = numpy.zeros((9,3))
#correction1[0,0:2], correction[0,2] = g.minimize()

modules = [7,10,13,16,12,14,15,17]
for i in range(8):
    g = Grid(pyFAI.detectors.Pilatus2M(),'grid1_wax_0004p.cbf',2.54,numpy.array([1030,734]),0.647)
    g.apply_threshold(level=10, module = modules[i])
    correction1[i+1,0:2], correction[i+1,2], = g.minimize()
