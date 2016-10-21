import os
import argparse as ap
import pprint as pp
from utils import download, unzip, res_dir

dataset_src = 'http://www.cs.fsu.edu/~liux/courses/' \
    'cap5415-2016/class-only/test1.zip'

if __name__ == '__main__':
    
    p = ap.ArgumentParser(
        prog='vis_proj',
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description='cli')
    
    p.add_argument(
        '-m', '--module',
        default='aeSIFT',
        help='choose a module to run.')
    
    p.add_argument(
        '-r', '--retrieve',
        default=False,
        action='store_true',
        help='retrieve web resources.')
    
    args = p.parse_args()
    
    if args.retrieve:
        print('Downloading: {}'.format(os.path.basename(dataset_src)))
        unzip(download(dataset_src))
    
    datapath = os.path.join(res_dir, 
        os.path.splitext(os.path.basename(dataset_src))[0])
    print('Dataset: {}'.format(datapath))
    
    