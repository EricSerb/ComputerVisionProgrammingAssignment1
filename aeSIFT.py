'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek


Used this as a reference:
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

'''
from utils import cmp_img, color_histo, color_thresh
from prSys import Manager
import cv2
import time

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
    
    
def mycomparer(a, b, qc):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    sift = cv2.SIFT()
    kp1, ds1 = sift.detectAndCompute(im1, None)
    kp2, ds2 = sift.detectAndCompute(im2, None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(ds1,ds2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1,0]
    
    draw_params = dict(
        matchColor = (0,255,0),
        singlePointColor = (255,0,0),
        matchesMask = matchesMask,
        flags = 0)
    
    img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
    
    plt.imshow(img3)
    plt.imsave('sometest.jpg')
    sys.stdin.readline()
    
    
def runtest(d):
    print('\naeSIFT test\n-----------')
    
    t = time.time()
    
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests()
    
    print(time.time() - t, 'sec')