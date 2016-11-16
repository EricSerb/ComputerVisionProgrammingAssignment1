'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

Driver for running the various segments of our project.
Run 'python main.py -h' for info.
'''
import argparse as ap
from time import time
from data import Dataset, f_join
from prSys import Manager
import aeCIBR as cibr
import aeSIFT as sift
import filteredCIBR as fcibr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

precfig = plt.figure()
rankfig = plt.figure()

def runtest(dset, mod):
    global precfig, rankfig
    
    t = time()
    print(str(mod.hdr))
    
    handle = mod.handler(dset)
    
    
    manage = Manager(dset, mod, cmp=handle)
    manage.alltests(dset.catsz, pfig=precfig, rfig=rankfig) # K
    
    with open('.'.join((mod.__name__, 'txt')), 'wb+') as fd:
        fd.write(str(mod.hdr) + '\n')
        for cat in dset.catlist: 
            fd.write('best = {} : {}\n'.format(
                cat, dset.cats[cat][manage.prs[cat].best[-1]].id))
            fd.write('wrst = {} : {}\n'.format(
                cat, dset.cats[cat][manage.prs[cat].wrst[-1]].id))
        fd.write('{} sec\n'.format(time() - t))
        
    print('\n time: {:0.04f} s\n'.format(time() - t))


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
        't' : ('tests', {
            'default' : 3,
            'help' : 'specify number of qry images to test',
        }),
    }
    
    for i in pargs:
        p.add_argument(
            '-{}'.format(i), '--{}'.format(pargs[i][0]), **pargs[i][1])
        
    # setup
    args = p.parse_args()
    dset = Dataset(args.src, args.dir, download=args.retrieve, cases=int(args.tests))
    
    modmap = { \
        cibr.__name__ : (('1', 'aecibr', 'cibr'), cibr), \
        fcibr.__name__ : (('2', 'filtered', 'filteredcibr', 'fcibr'), fcibr), \
        sift.__name__ : (('3', 'sift', 'aesift'), sift), \
    }
    
    if args.module is None:
        for mod in (cibr, fcibr, sift):
            runtest(dset, mod)
    else:
        foundmod = False
        for names in modmap:
            if args.module.lower() in modmap[names][0]:
                runtest(dset, modmap[names][1])
                foundmod = True
        if not foundmod:
            print('Module {} does not exist.'.format(args.module))
    
    plt.figure(precfig.number)
    plt.savefig(f_join(dset.dest, 'avg_Prec.jpg'))
    plt.close(precfig)
    plt.figure(rankfig.number)
    plt.savefig(f_join(dset.dest, 'avg_Rank.jpg'))
    plt.close(rankfig)
        
        
if __name__ == '__main__':
    main()