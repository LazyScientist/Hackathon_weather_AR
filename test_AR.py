import cv2
import numpy as np


frameheight = 480
framewidth = 640
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, 130)






def preProcessing(img):
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey,(5,5),1)
    imgCanny  = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel, iterations=2)
    imgthresh = cv2.erode(imgDial,kernel,iterations=1)

    return imgthresh



def getCountours(img):
    max_area = 0
    biggest = np.array([[[0,0]],[[1,1]],[[2,2]],[[3,3]]])
    countours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # retrival method, approximation method
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgcontour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area> max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv2.drawContours(imgcontour, biggest, -1, (255, 0, 0), 23)
            # print(len(approx))
            # objCor = len(approx)
            # x, y, w, h = cv2.boundingRect(approx)

    return biggest

def reorder(mypoints):
    mypoints = mypoints.reshape((4,2))
    mypointsnew = np.zeros((4,1,2),np.int32)
    add = mypoints.sum(1)

    mypointsnew[0] = mypoints[np.argmin(add)]
    mypointsnew[3] = mypoints[np.argmax(add)]

    diff = np.diff(mypoints,axis=1)
    mypointsnew[1] = mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]
    
    return mypointsnew



def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[framewidth,0],[0,frameheight],[framewidth,frameheight]])
    
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix,(framewidth,frameheight))

    imgcropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgcropped = cv2.resize(imgcropped,(framewidth,frameheight))

    return imgcropped


def augmentimg(biggest, img, imgAug):
    # biggest = reorder(biggest)

    h,w,c = imgAug.shape


    pts1 = np.array(biggest)
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    
    matrix, _ = cv2.findHomography(pts1,pts2)
    print((matrix*img.shape[0]/100) + img.shape[0] )
    
    imgOutput = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))

    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))

    imgOutput = img + imgOutput

    return imgOutput



while True:
    success , img = cap.read()
    img = cv2.resize(img, (framewidth, frameheight))
    
    imgAug = cv2.imread('yyy.jpg')
    # imgAug = cv2.resize(imgAug, (80, 80))



    imgcontour = img.copy()
    imgthresh = preProcessing(img)
    biggest = getCountours(imgthresh)
    print(biggest)
    # if biggest.size != 0:
        # imgwarped = getWarp(img,biggest)
        # cv2.imshow("warped", imgwarped)
    imaug = augmentimg(biggest,img, imgAug)
    cv2.imshow("Imagear",imaug)
    
    cv2.imshow('counter',imgcontour)
    cv2.imshow("orignal", imgthresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break