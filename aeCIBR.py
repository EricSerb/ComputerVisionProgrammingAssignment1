'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import cmp_img, color_histo, color_thresh, prec_rec
from operator import itemgetter
from pprint import pprint
import sys
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class PR(object):
    
    def __init__(self):
        self.best = None
        self.wrst = None
        self._avg = None
    
    def feed(self, pr):
        assert type(pr) == tuple
        assert len(pr) == 2
        p, r = pr
        print(p, r)
        self._update(p, r)
        # more
    
    def _update(self, P, R):
        pass
    
    def avg(self):
        return self._avg


def runtest(d):
    print('\naeCIBR test\n-----------')
    
    
    for tst_i in range(0,100):
        qry_classes = ('c1', 'c5', 'c9')
        queries = sorted(((c, d[c]) for c in d if c in qry_classes), key=itemgetter(1))
        tests = list((c, q[tst_i][1]) for (c, q) in queries)
        
        for t in tests:
            t_hist = color_histo(t[1])
            thresh = color_thresh(t_hist)
            norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) for t in t_hist)
            res = [('c' + str(k), [cmp_img(t_hist, i[1], norms, thresh) \
                for i in d['c' + str(k)]]) \
                for k in sorted(int(s[1:]) for s in d.keys())]
            
            P, R = zip(*[prec_rec(int(t[0][1:])-1, n, res) \
                for n in range(1, len(res[0][1]))])
            
            fig = plt.figure()
            print(dir(plt))
            plt.suptitle('plot')
            plt.subplot(121)
            plt.title('plot')
            plt.plot(P, R)
            plt.subplot(122)
            plt.scatter(P, R)
            plt.title('scatter')
            plt.savefig('myresult.jpg')
                # for r in res:
                    # print(r[0], r[1].count(True))
                # print(r[0], min(r[1]), max(r[1]), np.mean(r[1]))
            print(P)
            print(R)
            sys.stdin.readline()
            break
        break
        
