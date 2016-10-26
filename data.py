'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

General utilties module for the project that will be used to make module 
code cleaner and more readable, and mainly shorter :p
'''
import sys
import zipfile
from pprint import pprint
from random import sample
from getpass import getpass
from requests import get as wget
from requests.exceptions import RequestException
import numpy as np
from cv2 import imread
from utils import imgobj

from os import \
    makedirs as f_mkdir, \
    listdir as f_list, \
    getcwd as f_cwd

from os.path import \
    exists as f_exists, \
    join as f_join, \
    basename as f_base, \
    splitext as f_splitext


class Dataset(object):
    '''
    Encapsualtion of our dataset operations.
    '''
    def __init__(self, src, dest, \
        cases=3, download=False, catsize=100):
        
        self.src = src
        self.dest = dest
        self.catsz = int(catsize)
        self.cats = {}
        self.catlist = []
        self._dat = {}
        
        if isinstance(src, (bytes, str)) and download:
            f_path = self.download(src)
            self.unzip(f_path)
        
        self.rdir = f_join(dest, f_splitext(f_base(src))[0])
        self.f_path = f_join(self.rdir, 'image.orig')
        print('Dataset: {}'.format(self.f_path))
        
        self.files = []
        self.imgs = {}
        self.load()
        assert self.imgs
        
        
        self.testcases = None
        if type(cases) in (int, float):
            self.settests(int(cases))
        else:
            self.testcases = list(flatten(cases)) \
                if cases is not None else []
        
    
    
    def download(self, url):
        self.src = url
        print('Downloading: {}'.format(f_base(self.src)))
        if not f_exists(self.dest):
            f_mkdir(self.dest)
        name = f_join(self.dest, f_base(url))
        auth = (getpass('user: '), getpass('pswd: '))
        try:
            data = wget(url, auth=auth)
            with open(name, 'wb') as out:
                out.write(data.content)
        except RequestException:
            print('Fix your link')
            sys.exit(-1)
        return name
    
    
    def unzip(self, path):
        f = f_join(self.dest, 
            f_splitext(f_base(path))[0])
        print('Unzipping to: {}'.format(f))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(path=f, members=zf.namelist())
    
    
    def settests(self, n):
        assert type(n) == int
        n = 0 if n < 0 else n
        n = self.catsz if n > self.catsz else n
        self.testcases = self.testcases or \
            {qc : sample(range(0, self.catsz), n) for qc in self.catlist}
        pprint('Test cases:')
        pprint(self.testcases)
    
    
    def load(self):
        print('Loading images...')
        try:
            files = f_list(self.f_path)
            sfiles = sorted([int(f_splitext(f)[0]) for f in files])
            self.files = tuple('{}.jpg'.format(i) for i in sfiles)
            
            for f in self.files:
                
                p = f_join(f_cwd(), f_join(self.f_path, f))
                c = 'c{}'.format((int(f_splitext(f)[0]) // self.catsz) + 1)
                
                img = imgobj(f, c)
                self._dat[img.id] = imread(p)
                
                self.cats.setdefault(c, []).append(img)
                self.imgs[f] = img
                # putting everything in memory here
                # self.imgs.setdefault(c, []).append((f, imread(p)))
            
            print(list(c for c in self.cats))
            
            scat = sorted(int(c[1:]) for c in self.cats)
            self.catlist = ['c{}'.format(sc) for sc in scat]
            
        except (OSError, IOError) as e:
            print('Did you forget to download the data with \'-r\'?')
            print('Or maybe check your permissions?')
            raise e
        
        assert len(self.cats) == len(self.catlist) and \
            self.catsz == int(len(self.imgs) / len(self.cats)), \
            'Error: Category alignment.'
    
    
    def get(self, id):
        assert id in self._dat, 'Dataset.get: bad id'
        return self._dat[id]
    
    
    def idx(self, id):
        return int(f_splitext(id)[0]) % self.catsz
    
    
    def __iter__(self):
        '''
        Generator to walk over this data set in an ordered fashion.
        '''
        assert self.files
        for f in self.files:
            yield f
    