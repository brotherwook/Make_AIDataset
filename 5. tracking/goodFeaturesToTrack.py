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
        cv2.fillPoly(dil, contour, 255)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 10))
    dil = cv2.dilate(dil, k)

    contours, hierachy = cv2.findContours(dil, mode, method)

    for i, contour in enumerate(contours):
        cv2.fillPoly(dil, contour, 255)
        area = cv2.contourArea(contour)
        if 700 < area < 1300:
            cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
            print(area)


    cv2.imshow("frame", frame)
    cv2.imshow("thresh", dil)

    return dil


#%%
# LK용 변수
# lines = None  #추적 선을 그릴 이미지 저장 변수
prevImg = None
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
color = np.random.randint(0, 255, (200, 3))


#%%

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t += 1
    # 영상전처리 하지만 안씀
    # dil = imagepreprocessing(frame)

    # 처음 4프레임 데이터쌓기
    if t < 4:
        continue

        # LK시작

    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임 경우
    if prevImg is None:
        prevImg = gray
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
        # 추적 시작을 위한 코너 검출  ---①
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
    else:
        nextImg = gray
        # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg,
                                                       prevPt, None, criteria=termcriteria)
        # 대응점이 있는 코너, 움직인 코너 선별 ---③
        prevMv = prevPt[status == 1]
        nextMv = nextPt[status == 1]
        for i, (p, n) in enumerate(zip(prevMv, nextMv)):
            px, py = p.ravel()
            nx, ny = n.ravel()
            # 이전 코너와 새로운 코너에 선그리기 ---④
            cv2.line(lines, (px, py), (nx,ny), color[i].tolist(), 3)
            # 새로운 코너에 점 그리기
            cv2.circle(img_draw, (nx, ny), 2, color[i].tolist(), 2)
        # 누적된 추적 선을 출력 이미지에 합성 ---⑤
        frame = cv2.add(img_draw, lines)
        # 다음 프레임을 위한 프레임과 코너점 이월

        prevImg = nextImg
        prevPt = nextMv.reshape(-1, 1, 2)

    cv2.imshow("img_draw", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
