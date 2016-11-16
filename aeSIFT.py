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
<<<<<<< HEAD

# check cv2.__version__ == 3.1.0-dev
assert 'xfeatures2d' in dir(cv2), 'opencv tools missing: xfeatures2d (v3.0.1-dev)'

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
    def __init__(self, data, feats=5):
        '''
        Begin computation of descriptors and match trees.
        Uses 
        '''
        self.data = data
        self.ranks = {}
        self.descs = {}
        self.flann = {}
        assert isinstance(feats, int)
        self.sift = cv2.xfeatures2d.SIFT_create(feats)
        
        print('Calculating descriptors...')
        for img in data:
            self._getfeatures(img)
        
        print('Training matchers...')
        for c in data.catlist:
            frac = 1 # set this parameter to fraction of imgs to train with
            randimgs = sample(range(0, data.catsz), int(data.catsz * frac))
            self._train(c, [self.descs[data.cats[c][ri].id] for ri in randimgs])
        
        
    def _rank(self, qry, oth, r):
        self.ranks.setdefault(qry.id, []).append(Rank(oth, r))
=======
import time
from operator import itemgetter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# check cv2.__version__ == 3.1.0-dev
assert 'xfeatures2d' in dir(cv2), 'opencv tools missing: xfeatures2d'
# TODO still need to test this and stop the training so not always match class
debugging = False
saved_descs = {}
saved_flann = {}
sift = cv2.xfeatures2d.SIFT_create(10)


def mytrainer(qc, descs):
    global saved_flann
    print('Training {} with {} descriptors'.format(qc, len(descs)))
    
    saved_flann[qc] = cv2.FlannBasedMatcher(
        {'algorithm' : 1, 'trees' : 5},
        {'checks':30})
    
    saved_flann[qc].add(descs)
    saved_flann[qc].train()


def getfeatures(qc, img):
    global saved_descs, sift
    if qc not in saved_descs:
        saved_descs[qc] = {}
    kp, ds = sift.detectAndCompute(cv2.cvtColor(img[1], cv2.COLOR_BGR2HSV), None)
    saved_descs[qc][img[0]] = ds


def mycomparer(a, b, qc, qc2=None, best_matches=[]):
    global debugging, sift, saved_flann, saved_descs

    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    ds1 = saved_descs[qc][i1]
    ds2 = saved_descs[qc2][i2]

    bestCount = 0
    bestClass = None
    
    for c in saved_flann:
        
        # get matches for this flann matcher
        matches = saved_flann[c].knnMatch(ds2, ds1, k=2)
    
        # ratio test as per Lowe's paper
        matchCount = 0
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                matchCount += 1

    # store best matches based on match count
    if (not best_matches) or len(best_matches) < 5 and not any([i1 == x[0] for x in best_matches]):
        print('here', len(best_matches))
        best_matches.append((i1, matchCount))
        best_matches = sorted(best_matches, key=itemgetter(1), reverse=True)

    elif matchCount > best_matches[-1][1]:
        import sys
        print(best_matches)
        best_matches.pop()
        best_matches.append((i1, matchCount))
        # sort in reverse order to keep smallest at the back
        best_matches = sorted(best_matches, key=itemgetter(1), reverse=True)
        print(best_matches)
        sys.stdin.readline()

    else:
        # if this is reached then best_matches has 5 items and this pic did
        # not match as well as those pictures
        pass

    # keep track of best
    if matchCount > bestCount:
        bestCount, bestClass = matchCount, c

    return bestClass == qc


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
    print(manage.best_matches)
>>>>>>> 08e05ef4e28e76a973c28e9ffbc5a301e37629b2
    
    
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
        oth = oth[0]
        qry = qry[0]
        
        ds1 = self.descs[qry.id]
        
        rank = None
        
        best = Best(0, None)
        for c in self.flann:
            
            # get matches for this flann matcher
            matches = self.flann[c].knnMatch(ds1, k=2)
        
            # ratio test (Lowe)
            cnt = 0
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    cnt += 1
            
            # keep track of best
            if cnt > best.cnt:
                del best
                best = Best(cnt, c)
            
            # also remember result from flann associated with oth
            if c == oth.cat:
                rank = cnt
        
        self._rank(qry, oth, rank)
        
        return best.cat == oth.cat
