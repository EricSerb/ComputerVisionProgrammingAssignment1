'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
# from sortedcontainers.sortedlist import SortedList as slist
from utils import color_histo, color_thresh, Rank
import cv2


<<<<<<< HEAD
marker = 'o'
color = 'r'
marksz = 25
hdr = '\naeCIBR test\n-----------'

class handler(object):
    '''
    Handles the functions of a cibr system using 
    histogram matching, thresholding, etc.
    '''
    def __init__(self, dset, ifilter=None, gfilter=None):
        self.ifilter = ifilter
        self.gfilter = gfilter
        self.saved = {}
        self.dset = dset
        self.ranks = {}
=======
saved_cmp = {}
def mycomparer(a, b, qc, qc2, best_matches=[]):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
>>>>>>> 08e05ef4e28e76a973c28e9ffbc5a301e37629b2
    
    
    def _rank(self, qry, oth, r):
        self.ranks.setdefault(qry.id, []).append(Rank(oth, r))
    
    
    def __call__(self, oth, qry, ifilter=None, gfilter=None):
        '''
        Note: icat2 is note used here. It has beeb left as 
        we were using it for debugging purposes up to now.
        
        Break up the image nodes by name and file data.
        Check the object cache for hist, thresh, norm or generate/store.
        Use cmp function from utils module.
        If a filter has been set, it will be used.
        '''
        othdat = oth[-1]
        qrydat = qry[-1]
        oth = oth[0]
        qry = qry[0]
        
        # cache checking for histograms of both images
        if qry.id in self.saved:
            qhist, thresh, norms = self.saved[qry.id]
        else:
            qhist = color_histo(gfilter(ifilter(qrydat)) if (ifilter and gfilter) else qrydat)
            thresh = color_thresh(qhist)
            norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
                for t in qhist)
            self.saved[qry.id] = qhist, thresh, norms
        
        if oth.id in self.saved:
            ohist = self.saved[oth.id][0]
        else:
            ohist = color_histo(gfilter(ifilter(othdat)) if (ifilter and gfilter) else othdat)
            othresh = color_thresh(qhist)
            onorms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
                for t in qhist)
            self.saved[oth.id] = ohist, othresh, onorms
        
        
        res = []
        for (h1, h2, n, t) in zip(qhist, ohist, norms, thresh):
            res.append(cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT))
        
        self._rank(qry, oth, sum(res) / len(res))
        
        for i in range(len(res)):
            if (res[i] / norms[i]) < thresh[i]:
                return False
        return True
        
        