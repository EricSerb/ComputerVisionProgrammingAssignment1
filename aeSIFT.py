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
bestCount = {}
class_counts = {}
debugging = False
sift = cv2.xfeatures2d.SIFT_create()
curr_class = ''
def mycomparer(a, b, qc, qc2=None):
    global saved_cmp, bestCount, debugging, sift, curr_class
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    # check cv2.__version__ == 3.1.0-dev
    assert 'xfeatures2d' in dir(cv2), 'required opencv tools ' \
        'missing: xfeatures2d'
    
    # cache check and load if possible
    kp1 = ds1 = kp2 = ds2 = None
    if qc in saved_cmp:
        if i1 in saved_cmp[qc]:
            kp1, ds1 = saved_cmp[qc][i1]
    else:
        saved_cmp[qc] = {}
    if qc2 in saved_cmp:
        if i2 in saved_cmp[qc2]:
            kp2, ds2 = saved_cmp[qc2][i2]
    else:
        saved_cmp[qc2] = {}
        
    # load and fill cache if needed
    if kp1 is None:
        kp1, ds1 = sift.detectAndCompute(im1, None)
        saved_cmp[qc][i1] = kp1, ds1
    if kp2 is None:
        kp2, ds2 = sift.detectAndCompute(im2, None)
        saved_cmp[qc2][i2] = kp2, ds2
    
    # matching
    flann = cv2.FlannBasedMatcher({'algorithm' : 0, 'trees' : 5},{'checks':50})
    matches = flann.knnMatch(ds1,ds2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    matchCount = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.85 * n.distance:
            matchCount += 1
            matchesMask[i] = [1,0]
    
    # test drawing for debugging
    if debugging:
        draw_params = dict(
            matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = 0)
        
        img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
        plt.imshow(img3)
        plt.savefig('sometest.jpg')
        plt.close()
        
        print('i1:', i1, 'c1:', qc)
        print('i2:', i2, 'c2:', qc2)
        print('matches:', matchCount)
        print('---------------OK---------------')
    
        # print(bestCount)
        # bestCount.setdefault(qc, matchCount)
        if qc not in bestCount:
            bestCount[qc] = matchCount
        if matchCount > bestCount[qc]:
            bestCount[qc] = matchCount
    
    if qc != curr_class:
        print('i1:', i1, 'c1:', qc)
        print('i2:', i2, 'c2:', qc2)
        print('matches:', matchCount)
        print('---------------OK---------------')
        curr_class = qc
    
    if qc in class_counts:
        if qc == qc2:
            class_counts[qc].append(matchCount)
    else:
        class_counts[qc] = []
        if qc == qc2:
            class_counts[qc].append(matchCount)
    if i1 == '999.jpg' and i2 == '999.jpg':
        with open('counts.txt', 'ab+') as fd:
            for key in class_counts.keys():
                fd.write(key + ' ' + str(sum(class_counts[key])/len(class_counts[key])))
    return matchCount > 37 # lol
    
    
def runtest(d):
    print('\naeSIFT test\n-----------')
    
    t = time.time()
    
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests(N=100)
    # manage.alltests(qcs2plot=['c5'], N=100)
    
    print(time.time() - t, 'sec')