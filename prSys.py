'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

This module will contain the PR system used by each of the modules
in order to standardize how data is collected, and plots are generated, etc.
'''
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from data import f_exists, f_join, f_mkdir, res_dir


class PR(object):
    
    def __init__(self):
        self.save = []
        self.best = None
        self.wrst = None
        self.pavg = None
        self.n = 0
    
    def feed(self, p, r):
        assert type(p) == type(r) == float
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
    
import sys
class Manager(object):
    '''
    Manages pr calculations and maintains state for 
    avergaing, plotting, etc.
    '''
    def __init__(self, data, sub_f, cmp=(lambda a, b, c: True)):
        
        # get list of classes, and save other args
        self.data = data
        self.qry_classes = ['c{}'.format(i) \
            for i in sorted(int(c[1:]) for c in data)]
        self.sub = f_join(res_dir, sub_f)
        self.imgcmp = cmp
        
        # double check directories are set up
        if not f_exists(self.sub):
            f_mkdir(self.sub)
        for qc in self.qry_classes:
            sub_c = f_join(self.sub, qc)
            if not f_exists(sub_c):
                f_mkdir(sub_c)
        
        # grab PR objects for each qry class
        self.prs = {qc : PR() for qc in self.qry_classes}
    
    
    def test(self, qc, i, plot=True):
        '''
        Using the i'th img as the qry image, 
        run a PR test on class qc.
        '''
        assert qc in self.data
        assert i < len(self.data[qc])
        
        qi = self.data[qc][i]
        
        res = {c : [self.imgcmp(o, qi, qc) for o in self.data[c]] \
            for c in self.data}
        
        P, R = zip(*[self.calcPR(res, qc, n)
            for n in range(1, len(res[qc]))])
        
        # feed the running pr totals with the last PR
        # (i.e. the one ran for n == 100 images)
        self.prs[qc].feed(P[-1], R[-1])
        if plot:
            self.plotPR(qc, i, P, R)
    
    
    def calcPR(self, res, qc, n):
        n_crrct = float(res[qc][:n].count(True))
        n_mtchd = sum(res[j][:n].count(True) for j in res)
        try:
            p = n_crrct / n_mtchd
        except ZeroDivisionError:
            p = 0.0
            pass
        r = n_crrct / n
        return p, r
    
    
    def plotPR(self, qc, i, P, R):
        '''
        Handles making PR plot with mpl.
        '''
        assert type(P) == type(R)
        assert type(P) in (list, tuple)
        assert len(P) == len(R)
        
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
        
        plt.savefig(f_join(self.sub, qc, 
            '{:03d}_pr.jpg'.format(i)))
    
    
    def avgPR(self, fig=None, marker=None):
        '''
        After running, this function will generate PR plots
        for averages maintained by the PR objects.
        '''
        if fig is None:
            fig = plt.figure()
        if marker is None:
            marker = 'o'
        plt.suptitle('avg_PR')
        plt.subplot(111)
        plt.xlabel('Category')
        plt.ylabel('Avg precision')
        for i, qc in enumerate(self.qry_classes):
            plt.scatter(i, self.prs[qc].pavg, marker=marker)
        plt.xticks([i for i in range(10)], [c for c in self.qry_classes])
        plt.savefig(f_join(self.sub, 'avg_PR.jpg'))
        
    
    def fullPR(self, qc):
        fig = plt.figure()
        plt.suptitle('full_PR')
        plt.subplot(111)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        
        dat = self.prs[qc].save
        dl = len(dat)
        zp, zr = zip(*(dat))
        
        for (p, r, sz, c) in zip(\
            zp, zr, \
            np.linspace(10, 200, dl), \
            mpl.cm.rainbow(np.linspace(0, 1, dl))):
            plt.scatter(p, r, s=sz, color=c, alpha=0.5)
        
        plt.savefig(f_join(self.sub, qc, '_full_.jpg'))
        
    def alltests(self, qcs2plot=['c1', 'c5', 'c9'], N=2):
        assert N > 0 and N < 101
        for qc in self.qry_classes:
            for i in range(0, N):
                p = (qc in qcs2plot)
                self.test(qc, i, plot=False)
                
        # after all tests are run create the full PR plots
        # -- these use the PR results for every qry img,
        # -- but only taking the pr for n = 100 
        for qc in self.qry_classes:
            self.fullPR(qc)
        
        # we also print out the avg pr for each class 
        # on the same plot to compare the classes
        self.avgPR()
        
        