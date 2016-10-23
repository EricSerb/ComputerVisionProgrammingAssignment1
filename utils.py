'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

General utilties module for the project that will be used to make module 
code cleaner and more readable, and mainly shorter :p
'''
import sys
import cv2
import zipfile
from operator import itemgetter
from requests import get as wget
from getpass import getpass
import numpy as np

from os import \
    makedirs as f_mkdir, \
    listdir as f_list, \
    getcwd as f_cwd
from os.path \
    import exists as f_exists, \
    join as f_join, \
    basename as f_base, \
    splitext as f_splitext


color = ('b', 'g', 'r')
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
        print('Loading images...', end=' ')
        try:
            for f in f_list(self.imgs):
                p = f_join(f_cwd(), f_join(self.imgs, f))
                c = 'c{}'.format((int(f_splitext(f)[0]) // 100) + 1)
                # yes i know, we are putting everything in memory here
                # my system only goes up to about 6 GB during runtime
                data.setdefault(c, []).append((f, cv2.imread(p)))
            print('done.')
        except FileNotFoundError:
            print('Did you forget to download the data with \'-r\'?')
        return data


class Otsu(object):
    def __init__(self, hist):
        self.hist = hist
        self.q = hist.cumsum()
        self.thresh = 0
        self.I = len(hist)
        self.mf_min = np.inf
    
    def __iter__(self):
        bins = np.arange(self.I)
        for i in range(0, self.I):
            p1, p2 = np.hsplit(self.hist, [i])
            q1, q2 = self.q[i], self.q[-1] - self.q[i]
            i1, i2 = np.hsplit(bins, [i])
            if not q2 or not q1:
                continue
            u1, u2 = np.sum(p1 * i1) / q1, np.sum(p2 * i2) / q2
            s1, s2 = np.sum(((i1 - u1) ** 2) * p1) / q1, np.sum(((i2 - u2) ** 2) * p2) / q2
            
            yield i, s1 * q1 + s2 * q2
    
    def opt_thresh(self):
        for t, mf in self:
            self.thresh, self.mf_min = min(
                ((t, mf), (self.thresh, self.mf_min)), 
                key=itemgetter(1))
        return self.thresh


def grayscale_histo(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def color_thresh(hists):
    res = []
    for h in hists:
        res.append(Otsu(h.flatten()).opt_thresh() / 256)
    return res

def color_histo(img):
    nch = img.shape[-1]
    res = tuple(cv2.calcHist(img, [i], None, [256], [0, 256]) for i in range(nch))
    return res


def cmp_img(hist1, img2, norm, thresh):
    hist2 = color_histo(img2)
    assert len(hist1) == len(hist2) == len(norm) == len(thresh)
    res = tuple(True if (cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT) / n) > t else False \
        for (h1, h2, n, t) in zip(hist1, hist2, norm, thresh))
    return all(res)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 3, 5, 3)


# def box_filter(img):
    # return cv2.boxFilter(img, -1, 3)


def gaussian_filter(img):
    return cv2.GaussianBlur(img, 3, 1)


def prec_rec(class_idx, n, match_res):
    
    n_crrct = float(match_res[class_idx][1][:n].count(True))
    n_mtchd = sum(match_res[j][1][:n].count(True) 
        for j in range(len(match_res)))
    
    try:
        p = n_crrct / n_mtchd
    except ZeroDivisionError:
        p = 0.0
        pass
    r = n_crrct / n
    return p, r

