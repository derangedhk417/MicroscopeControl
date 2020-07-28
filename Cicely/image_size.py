import cv2
import matplotlib.pyplot as plt
import code
import sys
import numpy as np


def drawEdges(img):
    cvimg = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.uint8)
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    if cvimg is None:
        print('Image not found')
        exit()
        
    scale = np.sqrt((350**2) / (cvimg.shape[0] * cvimg.shape[1]))
    resized = cv2.resize(cvimg, (0, 0), fx=scale, fy=scale)
    denoisedimg = cv2.fastNlMeansDenoising(resized,30,30)
    plt.imshow(denoisedimg)
    plt.show()
    cannyimg = cv2.Canny(denoisedimg,12,82)
    plt.imshow(cannyimg)
    plt.show()
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
		(3, 3)
	)
    dilated = cv2.dilate(cannyimg, kernel)
    plt.imshow(dilated)
    plt.show()
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    min_area = 50
    edges = []

    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            edges.append(c)
        
    colorimg = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.uint8) 
    colorresized = cv2.resize(colorimg, (0, 0), fx=scale, fy=scale)
    contimg = cv2.drawContours(colorresized, edges, -1, (255, 0, 0))
    plt.imshow(contimg)
    plt.show()
    centroid(dilated, edges)

def centroid(img, contours):
    edges = []
    for c in contours:
        a = c.shape[0]
        b = c.shape[2]
        edges.append(c.reshape(a,b))
        
        
    imgs = []
    for i in range(len(edges)):
        edges[i] = np.int32([edges[i]])
        blank = np.zeros((img.shape[0],img.shape[1]))
        filledimg = cv2.fillPoly(blank, edges[i], 1).astype(np.uint8)
        plt.imshow(filledimg)
        plt.show()
        imgs.append(filledimg)
    
    moments = []
    for img in imgs:
        M = cv2.moments(img)
        moments.append(M)
        
    cXs = []
    cYs = []
    centroids = []
    for M in moments:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
        cXs.append(cX)
        cYs.append(cY)
        centroids.append(centroid)
        
    for img, centroid in zip(imgs, centroids):
        print(centroid)
        plt.imshow(img)
        plt.scatter(centroid[0],centroid[1])
        plt.show()
        
img = sys.argv[1]
drawEdges(img)

def imgSize():
    pass