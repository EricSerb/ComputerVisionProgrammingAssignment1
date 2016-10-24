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
import sys

import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


saved_flann = None
def mytrainer(qc, descs):
    global saved_flann
    print('Training {} with {} descriptors'.format(qc, len(descs)))
    if saved_flann is None:
        saved_flann = cv2.FlannBasedMatcher(
            {'algorithm' : 1, 'trees' : 5},
            {'checks':30})
    # for d in descs:
        # saved_flanns[qc].add(d)
    saved_flann.add(descs)
    saved_flann.train()
    
    
    
saved_cmp = {}
bestCount = {}
class_counts = {}
debugging = False
sift = cv2.xfeatures2d.SIFT_create()
curr_class = ''
curr_img = ''


saved_descs = {}
def getfeatures(qc, img):
    if qc not in saved_descs:
        saved_descs[qc] = {}
    kp, ds = sift.detectAndCompute(img[1], None)
    saved_descs[qc][img[0]] = ds
    # return ds


def mycomparer(a, b, qc, qc2=None):
    global saved_cmp, bestCount, debugging, sift, curr_class, curr_img, saved_flann, saved_descs
    i2, i1 = a[0], b[0]
    im2, im1 = a[1], b[1]
    
    # check cv2.__version__ == 3.1.0-dev
    assert 'xfeatures2d' in dir(cv2), 'required opencv tools ' \
        'missing: xfeatures2d'
    
    # cache check and load if possible
    # kp1 = ds1 = kp2 = ds2 = None
    # if qc in saved_cmp:
        # if i1 in saved_cmp[qc]:
            # kp1, ds1 = saved_cmp[qc][i1]
    # else:
        # saved_cmp[qc] = {}
    # if qc2 in saved_cmp:
        # if i2 in saved_cmp[qc2]:
            # kp2, ds2 = saved_cmp[qc2][i2]
    # else:
        # saved_cmp[qc2] = {}
        
    # load and fill cache if needed
    # if kp1 is None:
        # kp1, ds1 = sift.detectAndCompute(im1, None)
        # saved_cmp[qc][i1] = kp1, ds1
    # if kp2 is None:
        # kp2, ds2 = sift.detectAndCompute(im2, None)
        # saved_cmp[qc2][i2] = kp2, ds2
    
    # matching
    # flann = cv2.FlannBasedMatcher({'algorithm' : 0, 'trees' : 5},{'checks':50})
    # assert qc in saved_flanns
    
    ds2 = saved_descs[qc2][i2]
    ds1 = saved_descs[qc][i1]
    print('Calculating match: {} {} - {} {}'.format(qc, i1, qc2, i2))
    matches = saved_flann.knnMatch(ds1, ds2, k=2)
    print('Found: {}'.format(len(matches)))
    
    # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    matchCount = 0
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchCount += 1
            # matchesMask[i] = [1,0]
    print('Using: {}'.format(matchCount))
    
    # test drawing for debugging
    if debugging:
        # draw_params = dict(
            # matchColor = (0,255,0),
            # singlePointColor = (255,0,0),
            # matchesMask = matchesMask,
            # flags = 0)
        
        # img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,matches,None,**draw_params)
        # plt.imshow(img3)
        # plt.savefig('sometest.jpg')
        # plt.close()
        
        print('i1:', i1, 'c1:', qc)
        print('i2:', i2, 'c2:', qc2)
        print('matches:', matchCount)
        print('---------------OK---------------')
    
        # print(bestCount)
        # bestCount.setdefault(qc, matchCount)
        # if qc not in bestCount:
            # bestCount[qc] = matchCount
        # if matchCount > bestCount[qc]:
            # bestCount[qc] = matchCount
    
        if qc in class_counts:
            if qc == qc2:
                class_counts[qc].append(matchCount)
        else:
            class_counts[qc] = []
            if qc == qc2:
                class_counts[qc].append(matchCount)
        if i1 == '999.jpg' and i2 == '999.jpg':
            with open('counts.txt', 'ab+') as fd:
                for key in class_counts.keys():
                    fd.write(key + ' ' + str(sum(class_counts[key])/len(class_counts[key])))
    #### END DEBUGGING SECTION ####
        
    if qc != curr_class or i1 != curr_img:
        print('i1:', i1, 'c1:', qc)
        print('i2:', i2, 'c2:', qc2)
        print('matches:', matchCount)
        print('---------------OK---------------')
        curr_class = qc
        curr_img = i1
    
    
    print('len of descs for {} {}, {} {}:'.format(qc, i1, qc2, i2), len(saved_descs[qc][i1]), len(saved_descs[qc2][i2]))
    
    # matchThresh = min(len(saved_descs[qc][i]) for i in saved_descs[qc])
    
    matchThresh = 20
    print('Using thresh', matchThresh)
    # sys.stdin.readline()
    
    return matchCount >= matchThresh # lol
    
    
def runtest(d, cases, debug=False):
    global debugging
    debugging = debug
    print('\naeSIFT test\n-----------')
    t = time.time()
    
    print('Calculating descriptors...')
    for qc in d:
        for i in d[qc]:
            getfeatures(qc, i)
    print('done.')
    
    print('Training matchers...')
    for qc in d:
        mytrainer(qc, [saved_descs[qc][i[0]] for i in d[qc]])
    print('done.')
    
    manage = Manager(d, __name__, cmp=mycomparer)
    manage.alltests(N=100, pick3=cases)
    
    print(time.time() - t, 'sec')
    