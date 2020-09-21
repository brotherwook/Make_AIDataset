import numpy as np
import cv2
import argparse
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009140900.avi')
if (not cap.isOpened()):
    print('Error opening video')

#%% 그 외 전역변수
fgbg = cv2.createBackgroundSubtractorKNN()

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# 현재 프레임 번호 변수
t = 0

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

# 사람의 박스 크기
h_width = 20
h_height = 50


def imagepreprocessing(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    ret, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    medianblur = cv2.medianBlur(thresh, 5)

    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 10))
    dil = cv2.dilate(medianblur, k)

    contours, hierachy = cv2.findContours(dil, mode, method)

    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(dil, contour, 255)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 10))
    dil = cv2.dilate(dil, k)

    contours, hierachy = cv2.findContours(dil, mode, method)

    for i, contour in enumerate(contours):
        cv2.fillPoly(dil, contour, 255)
        area = cv2.contourArea(contour)
        if 700 < area < 900:
            x, y, width, height = cv2.boundingRect(contour)
            center = (int(x + width / 2), int(y + height / 2))
            cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)


    cv2.imshow("frame", frame)
    cv2.imshow("thresh", dil)

    return dil


#%% FAST 특징 검출기 생성
fast = cv2.FastFeatureDetector_create(100)

#%%

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t += 1

    dil = imagepreprocessing(frame)

    # 처음 4프레임 데이터쌓기
    if t < 4:
        continue

    # FAST 시작
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints = fast.detect(dil, None)
    frame = cv2.drawKeypoints(frame, keypoints, None)

    cv2.imshow("img_draw", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
