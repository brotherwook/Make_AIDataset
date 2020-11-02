import numpy as np
import cv2
import argparse
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%%
# 영상불러오기
cap = cv2.VideoCapture('D:/F20001_3_video/20201013/F20001_3_202010131200.avi')
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
color = np.random.randint(0, 255, (1000, 3))

temp = []
flag = 0
mask = None
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

    if flag == 0:
        while True:
            clone = frame.copy()
            cv2.imshow("image", frame)
            key = cv2.waitKey(0)

            if mask is None:
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = np.zeros_like(frame)

            if key == ord('r'):
                cv2.destroyAllWindows()
                clicked_points = []
                roi = [[[0, 341 * 2],
                        [0, 535 * 2],
                        [487 * 2, 520 * 2],
                        [738 * 2, 493 * 2],
                        [738 * 2, 410 * 2],
                        [387 * 2, 394 * 2],
                        [294 * 2, 386 * 2],
                        [290 * 2, 275 * 2],
                        [163 * 2, 266 * 2]]]
                flag = 1
                roi = np.array(roi)
                cv2.fillPoly(mask, roi, (255, 255, 255))
                temp.append(roi[0])
                print("roi 좌표:", roi)
                break

    if flag == 1:

        if len(roi[0]) > 0:
            # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
            # cv2.imshow("mask", mask)
            roi_img = cv2.bitwise_and(frame, mask)

            # roi 영역 빨간줄 긋기
            for j in range(len(temp)):
                for i, v in enumerate(roi[0]):
                    if i < len(roi[0]) - 1:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

        img_draw = roi_img.copy()
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        # 최초 프레임 경우
        if prevImg is None:
            prevImg = gray
            # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
            lines = np.zeros_like(frame)
            # 추적 시작을 위한 코너 검출  ---①
            prevPt = cv2.goodFeaturesToTrack(prevImg, 500, 0.01, 40)
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
                px, py = int(px), int(py)
                nx, ny = int(nx), int(ny)
                if px-2 <nx<px+2 and py-2 <ny<py+2:
                    continue
                # 이전 코너와 새로운 코너에 선그리기 ---④
                cv2.line(lines, (px, py), (nx,ny), color[i].tolist(), 3)
                # 새로운 코너에 점 그리기
                cv2.circle(img_draw, (nx, ny), 2, color[i].tolist(), 2)
                cv2.putText(img_draw,"person" + str(i), (nx,ny), cv2.FONT_HERSHEY_SIMPLEX,1, color[i].tolist(),2)
            # 누적된 추적 선을 출력 이미지에 합성 ---⑤
            frame = cv2.add(img_draw, lines)
            # 다음 프레임을 위한 프레임과 코너점 이월

            prevImg = nextImg
            prevPt = nextMv.reshape(-1, 1, 2)

        tmp = cv2.resize(frame, (int(width/2), int(height/2)))
        cv2.imshow("img_draw", tmp)

        k = cv2.waitKey(1)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
