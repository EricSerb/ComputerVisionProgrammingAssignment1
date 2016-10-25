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
assert 'xfeatures2d' in dir(cv2), 'opencv tools missing: xfeatures2d (v3.0.1-dev)'

hdr = '\naeSIFT test\n-----------'


class handler(object):
    '''
    Manages state for the sift system.
    At init we get/store descriptors and
    train a few look up tables (cv2.flann...).
    They are essentially KD trees that provide
    fast approximate nearest neighbor searching.
    '''
    def __init__(self, data):
        '''
        Begin computation of descriptors and match trees.
        Uses 
        '''
        self.ready = False
        self.debugging = False
        self.descs = {}
        self.flann = {}
        self.sift = cv2.xfeatures2d.SIFT_create(10)
        
        print('Calculating descriptors...')
        for icat in data.imgs:
            for i in data.imgs[icat]:
                self._getfeatures(icat, i)
        
        print('Training matchers...')
        for icat in data.imgs:
            self._train(icat, [self.descs[icat][i[0]] for i in data.imgs[icat]])
        
        self.data = data

    def _train(self, icat, descs):
        '''
        Loads the flann kd trees for specific category with descriptors.
        '''
        print('Training {} with {} descriptor sets'.format(icat, len(descs)))
        
        self.flann[icat] = cv2.FlannBasedMatcher(
            {'algorithm' : 1, 'trees' : 5},
            {'checks':30})
        
        self.flann[icat].add(descs)
        self.flann[icat].train()


    def _getfeatures(self, icat, img):
        '''
        Grabs a an image tuple from category icat.
        Uses cv2 SIFT to extract features from the images.
        Saves them in self.descs, a hash structure:
        
            { 'category' : {
                'img.jpg' : descriptors,
                },
            }
        '''
        if icat not in self.descs:
            self.descs[icat] = {}
        
        kp, ds = self.sift.detectAndCompute(
            cv2.cvtColor(img[1], cv2.COLOR_BGR2HSV), None)
        
        self.descs[icat][img[0]] = ds


    def __call__(self, a, b, icat, icat2):
        '''
        Uses each trained flann tree to compare the query image
        with. Each match set returned is filtered by a ratio test
        between two distances. We tuned this to 0.8 here.
        
            (m,n for m,n in matches if m.distance < 0.8 * n.distance)
        
        '''
        i2, i1 = a[0], b[0]
        im2, im1 = a[1], b[1]
        
        ds1 = self.descs[icat][i1]
        
        bestCount = 0
        bestClass = None
        
        for c in self.flann:
            
            # get matches for this flann matcher
            matches = self.flann[c].knnMatch(ds1, k=2)
        
            # ratio test (Lowe)
            matchCount = 0
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8 * n.distance:
                    matchCount += 1
            
            # keep track of best
            if matchCount > bestCount:
                bestCount, bestClass = matchCount, c
        
        return bestClass == icat2
