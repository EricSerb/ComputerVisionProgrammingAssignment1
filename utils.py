'''
Computer Vision - FSU - CAP5415
Adam Stallard, Eric Serbousek

General utilties module for the project that will be used to make module 
code cleaner and more readable, and mainly shorter :p
'''
import cv2
import zipfile
from requests import get as wget
from getpass import getpass
from os import makedirs as f_mkdir
from os.path \
    import exists as f_exists, \
    join as f_join, \
    basename as f_base, \
    splitext as f_splitext


color = ('b', 'g', 'r')
res_dir = 'res'


def download(url):
    if not f_exists(res_dir):
        f_makedir(res_dir)
    name = f_join(res_dir, f_base(url))
    auth = (getpass('user: '), getpass('pswd: '))
    data = wget(url, auth=auth)
    with open(name, 'wb') as out:
        out.write(data.content)
    return name


def unzip(path):
    f = f_join(res_dir, 
        f_splitext(f_base(path))[0])
    print('Unzipping to: {}'.format(f))
    with zipfile.ZipFile(path) as zf:
        for mem in zf.infolist():
            img = f_base(mem.filename)
            zf.extract(mem, f_join(f, img))


def grayscale_histo(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])


def color_histo(img):
    histos = []
    for i in range(len(color)):
        histos.append(cv2.calcHist([img], [i], None, [256], [0, 256]))
    return histos


def compare_histo(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)


def bilateral_filter(img):
    return cv2.bilateralFilter(img, 3, 5, 3)


def box_filter(img):
    return cv2.boxFilter(img, -1, 3)


def gaussian_filter(img):
    return cv2.GaussianBlur(img, 3, 1)
