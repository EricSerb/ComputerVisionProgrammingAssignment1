'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

Driver for running the various segments of our project.
Run 'python main.py -h' for info.
'''
import argparse as ap
from copy import deepcopy
from data import \
    Dataset, res_dir
import aeCIBR as cibr
import aeSIFT as sift
import filteredCIBR as fcibr


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
    # where { *args } is a dictionary of
    # keyword arguments used by p.parse_args
    #
        'm' : ('module', {
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
    if args.module is None:
        for mod in (cibr, sift, fcibr):
            mod.runtest(d)
    else:
        mod = args.module.lower()
        if mod in ('1', 'aecibr', 'cibr'):
            cibr.runtest(d)
        elif mod in ('2', 'filtered', 'filteredcibr', 'fcibr'):
            fcibr.runtest(d)
        elif mod in ('3', 'sift', 'aesift'):
            sift.runtest(d)
        else:
            print('Module {} does not exist.'.format(args.module))
            