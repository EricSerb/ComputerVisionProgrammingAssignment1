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
from random import sample
from utils import Best, Rank
import cv2

assert 'xfeatures2d' in dir(cv2), \
    'opencv tools missing: xfeatures2d (v3.0.1-dev)'

marker = '*'
color = 'b'
marksz = 25
hdr = '\naeSIFT test\n-----------'


class handler(object):
    '''
    Manages state for the sift system.
    At init we get/store descriptors and
    train a few look up tables (cv2.flann...).
    They are essentially KD trees that provide
    fast approximate nearest neighbor searching.
    '''
    def __init__(self, data, feats=50):
        '''
        Begin computation of descriptors and match trees.
        Uses 
        '''
        assert isinstance(feats, int)
        self.data = data
        self.descs = {}
        self.sift = cv2.xfeatures2d.SIFT_create(feats)
        self.flann = cv2.FlannBasedMatcher(
            {'algorithm' : 1, 'trees' : 5},
            {'checks':30})
        
        print('Calculating descriptors...')
        for img in data:
            self._getfeatures(img)
        
    def _getfeatures(self, img):
        '''
        Grabs a an image tuple from category icat.
        Uses cv2 SIFT to extract features from the images.
        Saves them in self.descs, a hash by img id.
        '''
        dat = self.data.get(img)
        kp, ds = self.sift.detectAndCompute(dat, None)
        self.descs[img] = ds
        
    def __call__(self, oth, qry):
        '''
        Uses each trained flann tree to compare the query image
        with. Each match set returned is filtered by a ratio test
        between two distances. We tuned this to 0.8 here.
        
            (m,n for m,n in matches if m.distance < 0.8 * n.distance)
        
        '''
        oth, qry = oth[0], qry[0]
        ds1, ds2 = self.descs[qry.id], self.descs[oth.id]
        matches = self.flann.knnMatch(ds1, ds2, k=2)
        
        # ratio test (Lowe)
        cnt = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                cnt += 1
        return cnt
        

        
        