import os
import cv2
import getpass
import zipfile
import requests

color = ('b', 'g', 'r')
res_dir = 'res'


def download(url):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    name = os.path.join(res_dir, os.path.basename(url))
    auth = ('cap5415', getpass.getpass())
    data = requests.get(url, auth=auth).content
    with open(name, 'wb') as out:
        out.write(data)
    return name


def unzip(path):
    folder = os.path.join(res_dir, 
        os.path.splitext(os.path.basename(path))[0])
    print('Unzipping to: {}'.format(folder))
    with zipfile.ZipFile(path) as zf:
        for mem in zf.infolist():
            img = os.path.basename(mem.filename)
            zf.extract(mem, os.path.join(folder, img))


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
