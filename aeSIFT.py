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
from collections import namedtuple as nt
from utils import color_histo, color_thresh
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
    def __init__(self, data, feats=10):
        '''
        Begin computation of descriptors and match trees.
        Uses 
        '''
        self.descs = {}
        self.flann = {}
        assert isinstance(feats, int)
        self.sift = cv2.xfeatures2d.SIFT_create(feats)
        
        print('Calculating descriptors...')
        for img in data:
            self._getfeatures(img)
        
        print('Training matchers...')
        for c in data.catlist:
            self._train(c, [self.descs[img.id] for img in data.cats[c]])
        
        self.data = data
        
        
    def _train(self, cat, descs):
        '''
        Loads the flann kd trees for specific category with descriptors.
        '''
        print('Training {: <4} with {} descriptor sets'.format(
            cat, len(descs)))
        
        self.flann[cat] = cv2.FlannBasedMatcher(
            {'algorithm' : 1, 'trees' : 5},
            {'checks':30})
        
        self.flann[cat].add(descs)
        self.flann[cat].train()
        
        
    def _getfeatures(self, img):
        '''
        Grabs a an image tuple from category icat.
        Uses cv2 SIFT to extract features from the images.
        Saves them in self.descs, a hash by img id.
        '''
        kp, ds = self.sift.detectAndCompute(
            cv2.cvtColor(img[1], cv2.COLOR_BGR2HSV), None)
        self.descs[img.id] = ds
        
        
    def __call__(self, oth, qry):
        '''
        Uses each trained flann tree to compare the query image
        with. Each match set returned is filtered by a ratio test
        between two distances. We tuned this to 0.8 here.
        
            (m,n for m,n in matches if m.distance < 0.8 * n.distance)
        
        '''
        ds1 = self.descs[oth.id]
        
        Best = nt('Best', 'cnt cat') # nample tuple
        best = Best(0, None)
        
        for c in self.flann:
            
            # get matches for this flann matcher
            matches = self.flann[c].knnMatch(ds1, k=2)
        
            # ratio test (Lowe)
            cnt = 0
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.8 * n.distance:
                    cnt += 1
            
            # keep track of best
            if cnt > best.cnt:
                del best
                best = Best(cnt, c)
        
        return best.cat == qry.cat
