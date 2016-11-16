'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import gaussian_filter, bilateral_filter
from aeCIBR import handler as cibr_handler

hdr = '\nfdCIBR test\n-----------'
marker = '+'
color = 'g'
marksz = 35

<<<<<<< HEAD
class handler(cibr_handler):
    '''
    Uses the basic cibr handler but with a filter.
    '''
    def __init__(self, dset):
        '''
        Inherit from aeCIBR.
        Ifilter arg is not a kwarg.
        Matcher calls super class always with filter on.
        '''
        super(self.__class__, self).__init__(dset, ifilter=bilateral_filter, gfilter=gaussian_filter)
=======
saved_cmp = {}
def mycomparer(a, b, qc, qc2, best_matches=[]):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    t_hist = thresh = norms = None
    if qc in saved_cmp:
        if i1 in saved_cmp[qc]:
            t_hist, thresh, norms = saved_cmp[qc][i1]
            
    if t_hist is None:
>>>>>>> 08e05ef4e28e76a973c28e9ffbc5a301e37629b2
        
    def __call__(self, o, q):
        '''
        See aeCIBR.handler.matcher().
        This forces the filter to the cibr matcher.
        '''
        return super(self.__class__, self).__call__(
            o, q, ifilter=self.ifilter, gfilter=self.gfilter)
    