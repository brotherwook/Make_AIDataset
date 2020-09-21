import numpy as np
import cv2

background = cv2.imread('C:/MyWorkspace/Detectioncode/temp/F18006_2_202009140900_bgr.png')
cap = cv2.VideoCapture('C:/MyWorkspace/Detectioncode/inputs/F18006_2/F18006_2_202009140900.avi')
if (not cap.isOpened()):
    print('Error opening video')

fgbg = cv2.createBackgroundSubtractorKNN()
newfgbg = cv2.createBackgroundSubtractorKNN()

#%%
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 0  # Binary Threshold
AREA_TH = 50  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

#%%
while True:
    ret, frame = cap.read()
    t += 1
    fgmask = fgbg.apply(frame)
    cv2.imshow("fgmask", fgmask)
    if t<10:
        # newfgmask = newfgbg.apply(background)
        continue
    ret, bImage = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    # 커널생성
    e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # 영상 전처리 침식, 팽창
    bImage = cv2.erode(bImage, e)
    bImage = cv2.dilate(bImage, k)

    # 1인부분(흰색부분) = 0이 아닌부분의 좌표값 저장
    nonzero = np.nonzero(bImage)

    new_img = background.copy()
    # cv2.imshow("back", background)

    # background에 객체부분만 값을 바꿔줌
    for y, x in zip(nonzero[0], nonzero[1]):
        new_img[y][x] = frame[y][x]

    cv2.imshow("new_img", new_img)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()