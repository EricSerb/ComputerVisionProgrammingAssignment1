'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import cmp_img, color_histo, color_thresh, prec_rec, f_exists, f_join, f_mkdir, res_dir
from operator import itemgetter
from pprint import pprint
import sys
import cv2
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class PR(object):
    
    def __init__(self):
        self.save = []
        self.best = None
        self.wrst = None
        self.pavg = None
        self.n = 0
    
    def feed(self, pr):
        assert type(pr) == tuple
        assert len(pr) == 2
        p, r = pr
        self._update(p, r)
    
    def _update(self, P, R):
        if not self.n:
            self.best = self.wrst = (P, R)
            self.pavg = P
        else:
            if (P*R) > (self.best[0] * self.best[1]):
                self.best = (P, R)
            if (P*R) < (self.wrst[0] * self.wrst[1]):
                self.wrst = (P, R)
            self.pavg = ((self.pavg * self.n) + P) / (self.n + 1)
        self.save.append((P, R))
        self.n += 1
    


def runtest(d):
    print('\naeCIBR test\n-----------')
    
    sub_f = f_join(res_dir, 'cibr')
    if not f_exists(sub_f):
        f_mkdir(sub_f)
    t1 = time.time()
    
    
    
    qry_classes = tuple('c{}'.format(i) for i in range(1, 11))
    for qc in qry_classes:
        sub_c = f_join(sub_f, qc)
        if not f_exists(sub_c):
            f_mkdir(sub_c)
                
    pr_totals = {qc : PR() for qc in qry_classes}
    
    for tst_i in range(0,100):
        # print(tst_i, end=' ')
        queries = sorted(((c, d[c]) for c in d if c in qry_classes), key=itemgetter(1))
        tests = list((c, q[tst_i][1]) for (c, q) in queries)
        
        for t in tests:
            t_hist = color_histo(t[1])
            thresh = color_thresh(t_hist)
            norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) for t in t_hist)
            res = [('c' + str(k), [cmp_img(t_hist, i[1], norms, thresh) \
                for i in d['c' + str(k)]]) \
                for k in sorted(int(s[1:]) for s in d.keys())]
            
            
            # for n in range(1, len(res[0][1])):
                # p, r = prec_rec(int(t[0][1:])-1, n, res)
                # print(p, r, t[0])
                # sys.exit()
            P, R = zip(*[prec_rec(int(t[0][1:])-1, n, res) \
                for n in range(1, len(res[0][1]))])
            
            pr_totals[t[0]].feed((P[-1], R[-1]))
            
            if tst_i < 1000 and t[0] in ('c1', 'c5', 'c9'):
                fig = plt.figure()
                plt.suptitle('PR')
                plt.subplot(111)
                plt.xlabel('Precision')
                plt.ylabel('Recall')
                for (p, r, sz, c) in zip(\
                    P, R, \
                    np.linspace(10, 200, len(P)), \
                    mpl.cm.rainbow(np.linspace(0, 1, len(P)))):
                    plt.scatter(p, r, s=sz, color=c, alpha=0.5)
                plt.plot(P, R, linestyle='--')
                plt.title('scatter')
                plt.savefig(f_join(sub_f, t[0], 
                    '{:03d}_myresult.jpg'.format(tst_i)))
    
    for qc in qry_classes:
        print(qc, pr_totals[qc].best, pr_totals[qc].wrst, pr_totals[qc].pavg)
    
        fig = plt.figure()
        plt.suptitle('full_PR')
        plt.subplot(111)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        
        dat = pr_totals[qc].save
        dl = len(dat)
        zp, zr = zip(*(dat))
        
        for (p, r, sz, c) in zip(\
            zp, zr, \
            np.linspace(10, 200, dl), \
            mpl.cm.rainbow(np.linspace(0, 1, dl))):
            plt.scatter(p, r, s=sz, color=c, alpha=0.5)

        
        plt.savefig(f_join(sub_f, qc, '_all_.jpg'))
        
    fig = plt.figure()
    plt.suptitle('avg_PR')
    plt.subplot(111)
    plt.xlabel('Category')
    plt.ylabel('Avg precision')
    
    for i, qc in enumerate(qry_classes):
        plt.scatter(i, pr_totals[qc].pavg)
    plt.xticks([i for i in range(10)], [qc for qc in qry_classes])
    plt.savefig(f_join(sub_f, 'avg_PR.jpg'))
    
    
    
    print(time.time() - t1)
