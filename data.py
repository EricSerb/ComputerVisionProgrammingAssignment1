'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

General utilties module for the project that will be used to make module 
code cleaner and more readable, and mainly shorter :p
'''
from cv2 import imread
import zipfile
from requests import get as wget
from getpass import getpass
import numpy as np

from os import \
    makedirs as f_mkdir, \
    listdir as f_list, \
    getcwd as f_cwd

from os.path import \
    exists as f_exists, \
    join as f_join, \
    basename as f_base, \
    splitext as f_splitext


res_dir = 'res'
src = 'http://www.cs.fsu.edu/~liux/courses/' \
        'cap5415-2016/class-only/test1.zip'


class Dataset(object):
    '''
    Encapsualtion of our dataset operations.
    '''
    def __init__(self, res=res_dir, src=src):
        self.src = src
        self.rdir = f_join(res, f_splitext(f_base(src))[0])
        self.imgs = f_join(self.rdir, 'image.orig')
        print('Dataset: {}'.format(self.rdir))
    
    def download(self, url):
        print('Downloading: {}'.format(f_base(self.src)))
        if not f_exists(res_dir):
            f_mkdir(res_dir)
        name = f_join(res_dir, f_base(url))
        auth = (getpass('user: '), getpass('pswd: '))
        data = wget(url, auth=auth)
        with open(name, 'wb') as out:
            out.write(data.content)
        return name
    
    def unzip(self, path):
        f = f_join(res_dir, 
            f_splitext(f_base(path))[0])
        print('Unzipping to: {}'.format(f))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(path=f, members=zf.namelist())
    
    def get(self):
        data = {}
        print('Loading images...')
        try:
            
            myfiles = f_list(self.imgs)
            sortedf = sorted([int(f_splitext(f)[0]) for f in myfiles])
            myfiles = ['{}.jpg'.format(i) for i in sortedf]
            
            for f in myfiles:
                p = f_join(f_cwd(), f_join(self.imgs, f))
                c = 'c{}'.format((int(f_splitext(f)[0]) // 100) + 1)
                # yes i know, we are putting everything in memory here
                # my system only goes up to about 6 GB during runtime
                data.setdefault(c, []).append((f, imread(p)))
            print('done.')
        except (OSError, IOError) as e:
            print('Did you forget to download the data with \'-r\'?')
            print('Or maybe check your permissions?')
            raise e
        return data
