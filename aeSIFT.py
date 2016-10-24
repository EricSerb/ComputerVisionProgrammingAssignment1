'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek


Used as reference:

http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

https://www.scivision.co/compiling-opencv3-with-extra-contributed-modules/

Note: The SIFT library calls here are a pain. They have moved around across 
each version of python/opencv.

We used python 2.7.12, with opencv versin 3.1.0-dev

If another version is used, this module likely will not work. Sry :(

'''
from utils import cmp_img, color_histo, color_thresh
from prSys import Manager
import cv2
import time
import sys

import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


saved_cmp = {}
def mycomparer(a, b, qc):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    t_hist = thresh = norms = None
    if qc in saved_cmp:
        if i1 in saved_cmp[qc]:
            t_hist, thresh, norms = saved_cmp[qc][i1]
            
    if t_hist is None:
        t_hist = color_histo(im1)
        thresh = color_thresh(t_hist)
        norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
            for t in t_hist)
        saved_cmp.setdefault(qc, {})[i1] = t_hist, thresh, norms
        
    return cmp_img(t_hist, im2, norms, thresh)
    
    
bestCount = {}
def mycomparer2(a, b, qc, qc2=None):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    
    # check cv2.__version__ == 3.1.0-dev
    assert 'xfeatures2d' in dir(cv2), 'required opencv tools ' \
        'missing: xfeatures2d'
        
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, ds1 = sift.detectAndCompute(im1, None)
        kp2, ds2 = sift.detectAndCompute(im2, None)
        
        flann = cv2.FlannBasedMatcher({'algorithm' : 0, 'trees' : 5},{'checks':50})
        matches = flann.knnMatch(ds1,ds2,k=2)
        
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        # ratio test as per Lowe's paper
        matchCount = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchCount += 1
                matchesMask[i] = [1,0]
        
        
        draw_params = dict(
            matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = 0)
        
        img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
        plt.imshow(img3)
        plt.savefig('sometest.png')
        
    else:
        print('Warning: your platform does not have ' \
            'required support for this module')
    
    # print(bestCount)
    # bestCount.setdefault(qc, matchCount)
    if qc not in bestCount:
        bestCount[qc] = matchCount
    if matchCount > bestCount[qc]:
        bestCount[qc] = matchCount
    
    print('i1:', i1, 'c1:', qc)
    print('i2:', i2, 'c2:', qc2)
    print('matches:', matchCount)
    print('---------------OK---------------')
    
    return matchCount > 4 # lol
    
    
def runtest(d):
    print('\naeSIFT test\n-----------')
    
    t = time.time()
    
    manage = Manager(d, __name__, cmp=mycomparer2)
    manage.alltests(qcs2plot=['c5'], N=2)
    
    print(time.time() - t, 'sec')