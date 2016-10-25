'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

Driver for running the various segments of our project.
Run 'python main.py -h' for info.
'''
import argparse as ap
from time import time
from data import Dataset
from prSys import Manager
import aeCIBR as cibr
import aeSIFT as sift
import filteredCIBR as fcibr



def runtest(dset, mod, dbg):
    
    t = time()
    
    if 'init' in dir(mod):
        print('Initializing module: {}'.format(mod.__name__))
        mod.init(dset)
    
    print(str(mod.hdr))
    manage = Manager(dset, mod.__name__, cmp=mod.matcher)
    manage.alltests(dset.catsz)
    
    with open('.'.join((mod.__name__, 'txt')), 'wb+') as fd:
        fd.write(str(mod.hdr) + '\n')
        for qc in dset.imgs: 
            fd.write('best = {} : {}\n'.format(
                qc, dset.imgs[qc][manage.prs[qc].best[-1]][0]))
            fd.write('wrst = {} : {}\n'.format(
                qc, dset.imgs[qc][manage.prs[qc].wrst[-1]][0]))
        fd.write('{} sec\n'.format(time() - t))
        
    print(time() - t)


def main():

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
        'g' : ('debug', {
            'default' : False,
            'action' : 'store_true',
            'help' : 'run in debug mode',
        }),
        'r' : ('retrieve', {
            'default' : False,
            'action' : 'store_true',
            'help' : 'retrieve web resources',
        }),
        's' : ('src', {
            'default' : 'http://www.cs.fsu.edu/~liux/courses/' \
                'cap5415-2016/class-only/test1.zip',
            'help' : 'specify url resource',
        }),
        'd' : ('dir', {
            'default' : 'res',
            'help' : 'specify results directory',
        }),
    }
    
    for i in pargs:
        p.add_argument(
            '-{}'.format(i), '--{}'.format(pargs[i][0]), **pargs[i][1])
        
    # setup
    args = p.parse_args()
    dset = Dataset(args.src, args.dir, download=args.retrieve, cases=3)
    
    
    modmap = { \
        cibr.__name__ : (('1', 'aecibr', 'cibr'), cibr), \
        fcibr.__name__ : (('2', 'filtered', 'filteredcibr', 'fcibr'), fcibr), \
        sift.__name__ : (('3', 'sift', 'aesift'), sift), \
    }
    
    foundmod = False
    if args.module is None:
        for mod in (cibr, fcibr, sift):
            runtest(dset, mod, args.debug)
    else:
        for names in modmap:
            if args.module.lower() in modmap[names][0]:
                runtest(dset, modmap[names][1], args.debug)
                foundmod = True
    if not foundmod:
        print('Module {} does not exist.'.format(args.module))
        
if __name__ == '__main__':
    main()