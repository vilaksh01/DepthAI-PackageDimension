import cv2 as cv
import numpy as np
# import depthai as dai
import streamlit as st
import random as rng

rng.seed(12345)

def edge_detection(inputImg):
    img_gray = cv.cvtColor(inputImg, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', img_gray)
    thresholdingImg = np.zeros((img_gray.shape[0], img_gray.shape[1], 1), np.int8)
    cv.imshow('thresholdBlank', thresholdingImg)
    # Gaussian Blur the input image
    # BlurImage = cv.GaussianBlur(src=img_gray, ksize=(5, 5), sigmaX=0, sigmaY=0, borderType=4)
    BlurImage = cv.bilateralFilter(img_gray, 9, 75, 75)
    cv.imshow('blur', BlurImage)
    # threshold generate
    CannyAccThresh = cv.threshold(src=BlurImage, thresh=0, maxval=255, type=cv.THRESH_OTSU, dst=thresholdingImg)
    print(CannyAccThresh)
    # Canny Edge detection
    CannyImage = cv.Canny(image=BlurImage, threshold1=25, threshold2=190, apertureSize=3)
    cv.imshow('canny', CannyImage)
    # Finding contours
    contours, hierarchy = cv.findContours(image=CannyImage, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
    print("contour and hierarchy")
    print(contours, hierarchy)

    # Draw contours
    drawing = np.zeros((CannyImage.shape[0], CannyImage.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)
    cv.waitKey(0)



img = cv.imread('/home/sumit/Desktop/box.jpg')
img = cv.resize(img, (500, 500))
edge_detection(img)
