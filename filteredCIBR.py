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

    def __call__(self, o, q):
        '''
        See aeCIBR.handler.matcher().
        This forces the filter to the cibr matcher.
        '''
        return super(self.__class__, self).__call__(
            o, q, ifilter=self.ifilter, gfilter=self.gfilter)
    