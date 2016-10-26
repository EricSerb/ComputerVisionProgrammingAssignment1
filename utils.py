'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

General utilties module for the project that will be used to make module 
code cleaner and more readable, and mainly shorter :p
'''
import cv2
import numpy as np
from operator import itemgetter
from collections import Iterable, namedtuple as nt



class imgobj(object):
    def __init__(self, id, cat):
        self.id, self.cat = id, cat

class Otsu(object):
    def __init__(self, hist):
        self.hist = hist
        self.q = hist.cumsum()
        self.I = len(hist)
        self.thresh = int(self.I / 2)
        self.mf_min = np.inf
        
    
    def __iter__(self):
        bins = np.arange(self.I)
        for i in range(0, self.I - 1):
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
        return float(self.thresh)


def grayscale_histo(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def color_thresh(hists):
    res = []
    for h in hists:
        res.append(Otsu(h.flatten()).opt_thresh() / 256)
    return res


def color_histo(img):
    nch = img.shape[-1]
    assert nch == 3
    bins = 56
    res = tuple(cv2.calcHist(img, [i], None, [bins], [0, 256]) for i in range(nch))
    # if 0 in res[0][0] or [0] in res[0][0]:
        # print res
    return res


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 3, 5, 3)


# def box_filter(img):
    # return cv2.boxFilter(img, -1, 3)


def gaussian_filter(img):
    return cv2.GaussianBlur(img, (3,3), 1)


def flatten(container):
    if isinstance(container, Iterable):
        for e in container:
            if isinstance(e, Iterable) and not isinstance(e, basestring):
                for sub in flatten(e):
                    yield sub
            else:
                yield e
    else:
        yield container
