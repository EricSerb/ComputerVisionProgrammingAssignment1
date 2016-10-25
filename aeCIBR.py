'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import cmp_img, color_histo, color_thresh
from prSys import Manager
import cv2
import time

hdr = '\naeCIBR test\n-----------'

class handler(object):
    '''
    Handles the functions of a cibr system using 
    histogram matching, thresholding, etc.
    '''
    def __init__(self, dset, ifilter=None):
        self.ifilter = ifilter if hasattr(ifilter, '__call__') else (lambda x: x)
        assert self.ifilter != -1, 'cibr handler: bad init argument: ifilter'
        self.saved_cmp = {icat : {} for icat in dset.imgs}
        self.dset = dset
    
    def __call__(self, iNode1, iNode2, icat1, icat2, ifilter=None):
        '''
        Note: icat2 is note used here. It has beeb left as 
        we were using it for debugging purposes up to now.
        
        Break up the image nodes by name and file data.
        Check the object cache for hist, thresh, norm or generate/store.
        Use cmp function from utils module.
        If a filter has been set, it will be used.
        '''
        i2, i1 = iNode1[0], iNode2[0]
        im2, im1 = iNode1[1], iNode2[1]
        
        t_hist = thresh = norms = None
        if icat1 in self.saved_cmp:
            if i1 in self.saved_cmp[icat1]:
                t_hist, thresh, norms = self.saved_cmp[icat1][i1]
                
        if t_hist is None:
        
            t_hist = color_histo(ifilter(im1) if ifilter else im1)
                
            thresh = color_thresh(t_hist)
            
            norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
                for t in t_hist)
                
            self.saved_cmp.setdefault(icat1, {})[i1] = t_hist, thresh, norms
            
        return cmp_img(t_hist, ifilter(im2) if ifilter else im2, norms, thresh)
        