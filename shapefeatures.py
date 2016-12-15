# -*- coding: utf-8 -*-
"""
Calculate morphological shape features from grayscale images.
Can either calculate features for an entire image, or split up an image into
a set of overlapping patches, calculating features for each.

Created on Fri Aug 30 15:20:49 2013

@author: John Quinn <jquinn@cit.ac.ug>
"""

import cv2
import ctypes
import numpy as np
import sys


def extract(img, attributes=None, filters=None, centiles=[10,30,50,70,90],
            momentfeatures=True):
    '''
    Extract a single set of features for a whole image.
    
    Attributes can be either None (default, calculate everything), or some
    list of the following integer values.
    
    0: Area
    1: Area of min. enclosing rectangle
    2: Square of diagonal of min. enclosing rectangle
    3: Cityblock perimeter
    4: Cityblock complexity (Perimeter/Area)
    5: Cityblock simplicity (Area/Perimeter)
    6: Cityblock compactness (Perimeter^2/(4*PI*Area))
    7: Large perimeter
    8: Large compactness (Perimeter^2/(4*PI*Area))
    9: Small perimeter
    10: Small compactness (Perimeter^2/(4*PI*Area))
    11: Moment of Inertia
    12: Elongation: (Moment of Inertia) / (area)^2
    13: Mean X position
    14: Mean Y position
    15: Jaggedness: Area*Perimeter^2/(8*PI^2*Inertia)
    16: Entropy
    17: Lambda-max (Max.child gray level - current gray level)
    18: Gray level
    
    Attributes are calculated for all the shapes in a max-tree, then converted
    into a fixed-size feature vector by taking the specified centiles.
        
    If momentfeatures is set to True, the output feature vector also contains
    moments of binary images obtained by thresholding the input image at
    different levels.
    '''
    features = _extract_hist(img, attributes, filters, centiles)
    if momentfeatures:
        momentfeatures = _extract_moment_features(img)
        features = np.hstack((features, momentfeatures))
    return features


def patchextract(img, size, step, attributes=None,
                 filters=None, centiles=[10,30,50,70,90],
                 momentfeatures=True):
    '''
    Split up an image into square overlapping patches, and return all
    features for each patch.
    '''
    height, width = img.shape
    features = []
    y = step
    while y < height:
        x = step
        while (x < width):
            left = x-(size/2)
            right = x+(size/2)
            top = y-(size/2)
            bottom = y+(size/2)
            patch = img[top:bottom, left:right]
            features.append(extract(patch, attributes, filters, centiles))
            x += step
        y += step

    return np.vstack(features)


def _extract_moment_features(imgray):
    '''
    Return a feature vector for a given image. Calculate a set of moments
    from binary images obtained by thresholding the input image at
    different levels.
    '''
    # How many gray levels to threshold at
    nthresholds = 6
    # Proportions of grayscale to begin and end thresholding
    lowthresh = 0.15
    highthresh = 0.45
    minmax = cv2.minMaxLoc(imgray)
    # Widen the range a bit to avoid errors in uniform image patches
    minpixel = minmax[0]
    maxpixel = minmax[1]

    if minpixel == maxpixel:
        minpixel = max(minpixel-1, 0)
        maxpixel = min(maxpixel+1, 255)

    startthresh = minpixel + (maxpixel-minpixel)*lowthresh
    stopthresh = minpixel + (maxpixel-minpixel)*highthresh
    stepthresh = (stopthresh-startthresh)/(nthresholds+1)
    thresholds = np.arange(startthresh+stepthresh, stopthresh, stepthresh)

    momentthresholds = thresholds[0:-1:2]
    nmomentthresholds = len(momentthresholds)

    c_f1 = np.zeros(nthresholds)
    c_f2 = np.zeros(nthresholds)
    c_f3 = np.zeros(nthresholds)

    m_f1 = np.zeros(nmomentthresholds)
    m_f2 = np.zeros(nmomentthresholds)
    m_f3 = np.zeros(nmomentthresholds)
    m_f4 = np.zeros(nmomentthresholds)
    m_f5 = np.zeros(nmomentthresholds)
    m_f6 = np.zeros(nmomentthresholds)
    m_f7 = np.zeros(nmomentthresholds)

    if (maxpixel-minpixel) > 10:

        for i in range(nthresholds):
            ### Contour features for every threshold ###
            thresh = thresholds[i]
            binarypatch = cv2.threshold(imgray, thresh, 255,
                                        cv2.THRESH_BINARY_INV)[1]
            contours, hierarchy = cv2.findContours(binarypatch,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            # find the biggest contour if there is more than one
            maxarea = -1
            if len(contours) > 0:
                for icontour in range(len(contours)):
                    area = cv2.contourArea(contours[icontour])
                    if area > maxarea:
                        maxarea = area

                c = contours[icontour]

                hull = cv2.convexHull(c, returnPoints=True)
                if len(hull) > 1:
                    hullarea = cv2.contourArea(hull)
                    # get features of the contour
                    c_f1[i] = maxarea

                    if hullarea > 0:
                        c_f2[i] = maxarea/hullarea
                        c_f3[i] = cv2.arcLength(c, closed=True)

            ### Moment features for subset of thresholds ###
            if thresh in momentthresholds:
                imomfeature = np.nonzero(momentthresholds == thresh)[0][0]

                binarypatch = cv2.threshold(imgray,thresh,255,
                                            cv2.THRESH_BINARY_INV)[1]
                m = cv2.moments(binarypatch, binaryImage=1)
                h = cv2.HuMoments(m)
                
                m_f1[imomfeature] = m['mu02']
                m_f2[imomfeature] = m['mu20']
                m_f3[imomfeature] = m['mu11']
                m_f4[imomfeature] = m['m00']
                m_f5[imomfeature] = h[0][0]
                m_f6[imomfeature] = h[1][0]
                m_f7[imomfeature] = h[2][0]

    featurevec = np.hstack(np.array([c_f1, c_f3, m_f1, m_f2, m_f3, m_f4, m_f5,
                    m_f6, m_f7]))

    return featurevec


def _extract_hist(img, attributes=None, filters=None,
                  centiles=[10,30,50,70,90]):
    '''
    Derive a single set of features from the morphological features of many
    connected components, by calulating centiles.
    '''
    nattributes = len(attributes)
    ncentiles = len(centiles)
    features = _extract_morphological_features(img, attributes, filters)
    histfeatures = np.zeros((nattributes, ncentiles))
    for a in range(nattributes):
        bagsize = features.shape[0]
        if bagsize > 1:
            histfeatures[a,:] = np.percentile(features[:,a], list(centiles))      
        elif bagsize == 1:
            histfeatures[a,:] = np.ones(len(centiles))*features[0,a]
    return histfeatures.ravel()


def _extract_morphological_features(img, attributes=None, filters=None,
                                    centiles=[25,50,75]):
    '''
    Find all the connected components at different threshold levels, and 
    extract shape a set of morphological features for each of them.
    '''
    assert(len(img.shape) == 2)
    assert(img.dtype == 'uint8')
    
    img2 = (255-img)/3
    
    features = np.zeros((1, len(attributes)))
    
    minmax = cv2.minMaxLoc(img2)
    if minmax[1]-minmax[0] > 5:
        if attributes is None:
            attributes = range(19)
        allfeatures = []
        for attr in attributes:
            f = _singleattribute(img2, attr)
            allfeatures.append(f)
        features = np.vstack(allfeatures).transpose()
        
        validinstances = np.ones(features.shape[0]) > 0
        if filters is not None:
            for filt in filters:
                att = filt[0]
                thresh = filt[2]
                if att in attributes:
                    idx = attributes.index(att)
                    if filt[1] == '<':
                        valid = features[:,idx] < thresh
                    else:
                        valid = features[:,idx] >= thresh                
                validinstances = np.logical_and(validinstances, valid)
            features = features[validinstances, :]
        
    return features


def _singleattribute(img, attribute):
    extractFeatures = ctypes.cdll.LoadLibrary('./libshapefeatures.so')
    extractFeatures.MaxTreeAttributes.restype = ctypes.POINTER(ctypes.c_float)
    extractFeatures.MaxTreeAttributes.argtype = [ctypes.c_int,
                                                 ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_ubyte),
                                                 ctypes.c_int,
                                                 ctypes.POINTER(ctypes.c_int)]
    count = (ctypes.c_int)(-1)
    imgvec = img.ravel()
    out = extractFeatures.MaxTreeAttributes(img.shape[0], img.shape[1],
                                            imgvec.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                            attribute,
                                            ctypes.byref(count))
    x = out[0:count.value]
    return np.array(x)


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    features = extract(img)
    print features
