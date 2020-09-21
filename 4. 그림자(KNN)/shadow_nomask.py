import numpy as np
import cv2

cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18003_2\F18003_2_202009140900.mp4')
if (not cap.isOpened()):
    print('Error opening video')

fgbg = cv2.createBackgroundSubtractorKNN()

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
    fgmask = fgbg.apply(frame)
    cv2.imshow("fgmask", fgmask)
    ret, bImage = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("b", bImage)

    contours, hierarchy = cv2.findContours(bImage, mode, method)
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > AREA_TH:
            # print(area)
            x, y, width, height = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()