'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import gaussian_filter
from aeCIBR import handler as cibr_handler

hdr = '\nfdCIBR test\n-----------'


class handler(cibr_handler):
    '''
    Uses the basic cibr handler but with a filter.
    '''
    def __init__(self, dset, ifilter):
        '''
        Inherit from aeCIBR.
        Ifilter arg is not a kwarg.
        Matcher calls super class always with filter on.
        '''
        assert hasattr(ifilter, '__call__'), \
            'cibr handler: bad init argument: ifilter'
        self.ifilter = ifilter
        self.saved_cmp = {icat : {} for icat in dset.imgs}
        self.dset = dset
        
    def __call__(self, a, b, qc, qc2):
        '''
        See aeCIBR.handler.matcher().
        This forces the filter to the cibr matcher.
        '''
        return super(self.__class__, self).__call__(
            a, b, qc, qc2, ifilter=self.ifilter)
    