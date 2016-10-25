'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import cmp_img, color_histo, color_thresh
from prSys import Manager
import cv2
import time

hdr = '\naeCIBR test\n-----------'
saved_cmp = {}

def init(dset):
    global saved_cmp
    saved_cmp = {qc : {} for qc in dset.imgs}

def matcher(a, b, qc, qc2, ifilter=None):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    t_hist = thresh = norms = None
    if qc in saved_cmp:
        if i1 in saved_cmp[qc]:
            t_hist, thresh, norms = saved_cmp[qc][i1]
            
    if t_hist is None:
    
        t_hist = color_histo(ifilter(im1) if ifilter else im1)
            
        thresh = color_thresh(t_hist)
        
        norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
            for t in t_hist)
            
        saved_cmp.setdefault(qc, {})[i1] = t_hist, thresh, norms
        
    return cmp_img(t_hist, ifilter(im2) if ifilter else im2, norms, thresh)
    