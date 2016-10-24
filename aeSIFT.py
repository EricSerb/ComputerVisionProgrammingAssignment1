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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# check cv2.__version__ == 3.1.0-dev
assert 'xfeatures2d' in dir(cv2), 'opencv tools missing: xfeatures2d'


debugging = False
saved_descs = {}
saved_flann = {}
sift = cv2.xfeatures2d.SIFT_create()


def mytrainer(qc, descs):
    global saved_flann
    print('Training {} with {} descriptors'.format(qc, len(descs)))
    
    saved_flann[qc] = cv2.FlannBasedMatcher(
        {'algorithm' : 1, 'trees' : 5},
        {'checks':30})
    
    saved_flann[qc].add(descs)
    # saved_flann[qc].train()


def getfeatures(qc, img):
    global saved_descs, sift
    if qc not in saved_descs:
        saved_descs[qc] = {}
    kp, ds = sift.detectAndCompute(img[1], None)
    saved_descs[qc][img[0]] = ds


def mycomparer(a, b, qc, qc2=None):
    global debugging, sift, saved_flann, saved_descs
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    ds1 = saved_descs[qc][i1]
    
    bestCount = 0
    bestClass = None
    
    for c in saved_flann:
        
        # get matches for this flann matcher
        matches = saved_flann[c].knnMatch(ds1, k=2)
    
        # ratio test as per Lowe's paper
        matchCount = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                matchCount += 1
        
        # keep track of best
        if matchCount > bestCount:
            bestCount, bestClass = matchCount, c
    
    return bestClass == qc2


def runtest(d, cases, debug=False):
    global debugging
    debugging = debug
    print('\naeSIFT test\n-----------')
    t = time.time()
    
    print('Calculating descriptors...')
    for qc in d:
        for i in d[qc]:
            getfeatures(qc, i)
    
    print('Training matchers...')
    for qc in d:
        mytrainer(qc, [saved_descs[qc][i[0]] for i in d[qc]])
    
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests(N=100, pick3=cases)
    
    print(time.time() - t, 'sec')
    