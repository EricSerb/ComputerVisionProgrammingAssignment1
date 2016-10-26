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
from data import f_exists, f_join, f_mkdir

# __debug__ = False

class PR(object):
    
    def __init__(self):
        self.best = None
        self.wrst = None
        self.pavg = []
        self.ravg = []
        self.n = 0

    def feed(self, P, R, i):
        if not self.n:
            self.best = self.wrst = (P[-1], R[-1], i)
        else:
            if (P[-1]*R[-1]) > (self.best[0] * self.best[1]):
                self.best = (P[-1], R[-1], i)
            if (P[-1]*R[-1]) < (self.wrst[0] * self.wrst[1]):
                self.wrst = (P[-1], R[-1], i)
        self.pavg.append(sum(P) / float(len(P)))
        self.ravg.append(sum(R) / float(len(R)))
        self.n += 1
    
    
class Manager(object):
    '''
    Manages pr calculations and maintains state for 
    avergaing, plotting, etc.
    '''
    def __init__(self, data, sub_f, cmp=(lambda a, b, c: True)):
        
        # get list of classes, and save other args
        self.data = data
        self.sub = f_join(data.dest, sub_f)
        self.imgcmp = cmp
        
        # double check directories are set up
        if not f_exists(self.sub):
            f_mkdir(self.sub)
        for c_dir in self.data.catlist:
            sub_c = f_join(self.sub, c_dir)
            if not f_exists(sub_c):
                f_mkdir(sub_c)
        
        # grab PR objects for each qry class
        self.prs = {cat : PR() for cat in self.data.catlist}
    
    
    def test(self, cat, i, plot=True):
        '''
        Using the i'th img as the qry image, 
        run a PR test on class cat.
        '''
        assert cat in self.data.catlist
        assert i < len(self.data.cats[cat])
        
        qry = self.data.cats[cat][i]
        qry = [qry, self.data.get(qry.id)]
        
        res = {}
        for oth in self.data:
            o = self.data.imgs[oth]
            res.setdefault(o.cat, []).append(self.imgcmp([o, self.data.get(o.id)], qry))

        P, R = zip(*[self.calcPR(res, cat, n)
            for n in range(1, len(res[cat]) + 1)])
        
        # feed the running pr totals with the last PR
        # (i.e. the one ran for n == 100 images)
        self.prs[cat].feed(P, R, i)
        if plot:
            self.plotPR(cat, i, P, R)
    
    
    def calcPR(self, res, cat, k):
        '''
        Internal method for extracting P,R values. 
        res = container storing results by class and img within.
        cat = class of which to calc from.
        k = number of imgs from each class to retrieve.
        '''
        n_crrct = float(res[cat][:k].count(True))
        n_mtchd = float(sum(res[j][:k].count(True) for j in res))
        try:
            p = n_crrct / n_mtchd
        except ZeroDivisionError:
            p = 0.0
            pass
        r = n_crrct / k
        return p, r
    
    
    def plotPR(self, cat, i, P, R):
        '''
        Handles making simple PR plot with mpl.
        Data came from test results using qry image at data.imgs[cat][i].
        '''
        assert type(P) == type(R)
        assert type(P) in (list, tuple)
        assert len(P) == len(R)
        
        fig = plt.figure()
        plt.suptitle('PR')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        
        for (p, r, sz, c) in zip(\
            P, R, \
            np.linspace(10, 200, len(P)), \
            mpl.cm.rainbow(np.linspace(0, 1, len(P)))):
            plt.scatter(p, r, s=sz, color=c, alpha=0.5)
        
        plt.plot(P, R, linestyle='--')
        
        plt.savefig(f_join(self.sub, cat, 
            '{:03d}_pr.jpg'.format(i)))
        plt.close(fig)
    
    
    def avgPR(self):
        '''
        After running, this function will generate PR plots
        for averages maintained by the PR objects.
        '''
        fig = plt.figure()
        plt.suptitle('avg_PR')
        plt.title('prec = o, rec = x')
        plt.xlabel('Category')
        plt.xticks([i for i in range(10)], [c for c in self.data.catlist])
        plt.ylabel('Avg precision')
        plt.ylim(0.0, 1.1)
        plt.grid(True)
        
        for i, cat in enumerate(self.data.catlist):
            plt.scatter(i, sum(self.prs[cat].pavg) / len(self.prs[cat].pavg), 
                s=10, color='g', marker='o', alpha=0.6)
            plt.scatter(i, sum(self.prs[cat].ravg) / len(self.prs[cat].ravg), 
                s=20, color='b', marker='x', alpha=0.8)
        
        plt.savefig(f_join(self.sub, 'avg_PR.jpg'))
        plt.close(fig)
        
    
    def fullPR(self, cat):
        '''
        Plots all the weighted averaged precisions from all tests run 
        on each class.
        '''
        fig = plt.figure()
        plt.suptitle('full_PR')
        plt.xlabel('Precision')
        plt.xlim(0, 1.5)
        plt.ylabel('Recall')
        plt.ylim(0, 1.5)
        
        pavg = self.prs[cat].pavg
        ravg = self.prs[cat].ravg
        dl = len(pavg)
        
        for (p, r, sz, c) in zip(\
            pavg, ravg, np.linspace(10, 200, dl), \
            mpl.cm.rainbow(np.linspace(0, 1, dl))):
            
            plt.scatter(p, r, s=sz, color=c, alpha=0.5)
        
        plt.savefig(f_join(self.sub, cat, '_full_.jpg'))
        plt.close(fig)
        
        
        
    def plotRanks(self, id):
        ranks = self.cmp.topranks
        for imgid in ranks:
            pass
    
    
    def alltests(self, K, plots=['c1', 'c5', 'c9']):
        '''
        Executes all tests in an ordered fashion.
        '''
        assert isinstance(K, int)
        assert K > 0 and K < self.data.catsz + 1, \
            'Invalid K value. Must be in [0..{}]'.format(self.data.catsz)
        
        
        for c in self.data.catlist:
            print(' testing {:.<9}'.format(c))
            for i in self.data.testcases[c]:
                self.test(c, i, plot=(c in plots))
                
        # after all tests are run create the full PR plots
        # -- these use the PR results for every qry img,
        # -- but only taking the pr for n = 100 
        for cat in self.data.catlist:
            self.fullPR(cat)
        
        # we also print out the avg pr for each class 
        # on the same plot to compare the classes
        self.avgPR()
        
        
        