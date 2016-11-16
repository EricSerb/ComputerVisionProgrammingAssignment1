'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek


Used as reference:

http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

https://www.scivision.co/compiling-opencv3-with-extra-contributed-modules/

Note: The SIFT library calls here are a pain. They have moved around across 
each version of python/opencv.

We used python 2.7.12, with opencv versin 3.1.0-dev

If another version is used, this module likely will not work. Sry :(

'''
from utils import cmp_img, color_histo, color_thresh
from prSys import Manager
import cv2
import time
from operator import itemgetter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# check cv2.__version__ == 3.1.0-dev
assert 'xfeatures2d' in dir(cv2), 'opencv tools missing: xfeatures2d'
# TODO still need to test this and stop the training so not always match class
debugging = False
saved_descs = {}
saved_flann = {}
sift = cv2.xfeatures2d.SIFT_create(10)


def mytrainer(qc, descs):
    global saved_flann
    print('Training {} with {} descriptors'.format(qc, len(descs)))
    
    saved_flann[qc] = cv2.FlannBasedMatcher(
        {'algorithm' : 1, 'trees' : 5},
        {'checks':30})
    
    saved_flann[qc].add(descs)
    saved_flann[qc].train()


def getfeatures(qc, img):
    global saved_descs, sift
    if qc not in saved_descs:
        saved_descs[qc] = {}
    kp, ds = sift.detectAndCompute(cv2.cvtColor(img[1], cv2.COLOR_BGR2HSV), None)
    saved_descs[qc][img[0]] = ds


def mycomparer(a, b, qc, qc2=None, best_matches=[]):
    global debugging, sift, saved_flann, saved_descs

    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    ds1 = saved_descs[qc][i1]
    ds2 = saved_descs[qc2][i2]

    bestCount = 0
    bestClass = None
    
    for c in saved_flann:
        
        # get matches for this flann matcher
        matches = saved_flann[c].knnMatch(ds2, ds1, k=2)
    
        # ratio test as per Lowe's paper
        matchCount = 0
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                matchCount += 1

    # store best matches based on match count
    if (not best_matches) or len(best_matches) < 5 and not any([i1 == x[0] for x in best_matches]):
        print('here', len(best_matches))
        best_matches.append((i1, matchCount))
        best_matches = sorted(best_matches, key=itemgetter(1), reverse=True)

    elif matchCount > best_matches[-1][1]:
        import sys
        print(best_matches)
        best_matches.pop()
        best_matches.append((i1, matchCount))
        # sort in reverse order to keep smallest at the back
        best_matches = sorted(best_matches, key=itemgetter(1), reverse=True)
        print(best_matches)
        sys.stdin.readline()

    else:
        # if this is reached then best_matches has 5 items and this pic did
        # not match as well as those pictures
        pass

    # keep track of best
    if matchCount > bestCount:
        bestCount, bestClass = matchCount, c

    return bestClass == qc


def runtest(d, cases, debug=False):
    global debugging
    debugging = debug
    print('\naeSIFT test\n-----------')
    t = time.time()
    
    print('Calculating descriptors...')
    for qc in d:
        for i in d[qc]:
            getfeatures(qc, i)
    
    print('Training matchers...')
    for qc in d:
        mytrainer(qc, [saved_descs[qc][i[0]] for i in d[qc]])
    
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests(N=100, pick3=cases)
    print(manage.best_matches)
    
    with open('.'.join((__name__, 'txt')), 'wb+') as fd:
        fd.write('start_________________')
        fd.write(__name__ + '\n')
        for qc in d:
            p1 = 'best = ' + qc + ' : ' + d[qc][manage.prs[qc].best[-1]][0] + '\n'
            p2 = 'worst = ' + qc + ' : ' + d[qc][manage.prs[qc].wrst[-1]][0] + '\n'
            fd.write(p1)
            fd.write(p2)
        fd.write('end_________________')
    
    print(time.time() - t, 'sec')
