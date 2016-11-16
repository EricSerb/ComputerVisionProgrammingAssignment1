'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

'''
from utils import cmp_img, color_histo, color_thresh
from prSys import Manager
import cv2
import time


saved_cmp = {}
def mycomparer(a, b, qc, qc2, best_matches=[]):
    
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    t_hist = thresh = norms = None
    if qc in saved_cmp:
        if i1 in saved_cmp[qc]:
            t_hist, thresh, norms = saved_cmp[qc][i1]
            
    if t_hist is None:
        t_hist = color_histo(im1)
        thresh = color_thresh(t_hist)
        norms = tuple(cv2.compareHist(t, t, cv2.HISTCMP_INTERSECT) \
            for t in t_hist)
        saved_cmp.setdefault(qc, {})[i1] = t_hist, thresh, norms
        
    return cmp_img(t_hist, im2, norms, thresh)
    
    
def runtest(d, cases, debug=False):
    print('\naeCIBR test\n-----------')
    t = time.time()
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests(pick3=cases)

    with open('.'.join((__name__, 'txt')), 'wb+') as fd:
        fd.write('start_________________')
        fd.write(__name__+'\n')
        for qc in d:
            p1 = 'best = ' + qc + ' : ' + d[qc][manage.prs[qc].best[-1]][0] + '\n'
            p2 = 'worst = ' + qc + ' : ' + d[qc][manage.prs[qc].wrst[-1]][0] + '\n'
            fd.write(p1)
            fd.write(p2)
        fd.write('end_________________' + '\n')
    
    print(time.time() - t, 'sec')
