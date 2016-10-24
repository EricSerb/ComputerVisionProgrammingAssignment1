'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import cmp_img, color_histo, color_thresh, gaussian_filter
from prSys import Manager
import cv2
import time


saved_cmp = {}
def mycomparer(a, b, qc, qc2):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    t_hist = thresh = norms = None
    if qc in saved_cmp:
        if i1 in saved_cmp[qc]:
            t_hist, thresh, norms = saved_cmp[qc][i1]
            
    if t_hist is None:
        
        t_hist = color_histo(gaussian_filter(im1))
        thresh = color_thresh(t_hist)
        norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
            for t in t_hist)
        saved_cmp.setdefault(qc, {})[i1] = t_hist, thresh, norms
        
    return cmp_img(t_hist, gaussian_filter(im2), norms, thresh)
    


def runtest(d):
    print('\nfilteredCIBR test\n-----------')
    
    t = time.time()
    
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests()
    
    print(time.time() - t, 'sec')
    
