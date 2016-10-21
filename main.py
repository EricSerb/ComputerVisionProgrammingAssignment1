import argparse as ap
import pprint as pp

if __name__ == '__main__':
    
    p = ap.ArgumentParser(
        prog='vis_proj',
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description='cli')
    
    p.add_argument(
        '-m', '--module',
        default='asSIFT',
        help='choose a module to run.')
    
    args = p.parse_args()
    print('Running', args.module)
    