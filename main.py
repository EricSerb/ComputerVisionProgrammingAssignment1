'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

Driver for running the various segments of our project.
Run 'python main.py -h' for info.
'''
import sys
import argparse as ap
import pprint as pp
from os.path import join, basename, splitext
from copy import deepcopy
from utils import \
    Dataset, res_dir, \
    f_join, f_base, f_splitext
import aeCIBR as cibr
import aeSIFT as sift
import filteredCIBR as fcibr

from utils import f_join


if __name__ == '__main__':

    p = ap.ArgumentParser(
        prog='vis_proj',
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description='cli')
    
    pargs = {
    #
    # Add arg in this format:
    #
    #   'short' : ('long', { *args }),
    #
    # where *args is a dictionary of
    # keyword arguments used by p.parse_args
    #
        'm' : ('module', {
            'default' : 'aeSIFT',
            'help' : 'choose a module to run',
        }),
        'r' : ('retrieve', {
            'default' : False,
            'action' : 'store_true',
            'help' : 'retrieve web resources',
        }), 
    }
    
    # configure
    src = 'http://www.cs.fsu.edu/~liux/courses/' \
        'cap5415-2016/class-only/test1.zip'
    for i in pargs:
        p.add_argument(
            '-{}'.format(i), # short
            '--{}'.format(pargs[i][0]),  # long
            **pargs[i][1]) # *args
    
    # setup
    args = p.parse_args()
    data = Dataset(res=res_dir, src=src)
    if args.retrieve:
        data.unzip(data.download(src))
    
    # tests
    d = data.get()
    
    for mod in (cibr, sift, fcibr):
        mod.runtest(deepcopy(d))
        
        