import cv2


color = ('b', 'g', 'r')


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
