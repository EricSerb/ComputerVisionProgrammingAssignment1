'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import gaussian_filter
from aeCIBR import matcher as cibrmatcher, init as cibrinit

hdr = '\nfdCIBR test\n-----------'

def init(d):
    cibrinit(d)
    
def matcher(a, b, qc, qc2):
    return cibrmatcher(a, b, qc, qc2, ifilter=gaussian_filter)
    