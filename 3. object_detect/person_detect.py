import cv2
import numpy as np

#%%
background_path = "C:/MyWorkspace/Detectioncode/temp/F18006_2_202009111900.png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009111900.avi')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 50  # Binary Threshold
AREA_TH = 30  # Area Threshold

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

while True:
    try:
        retval, frame = cap.read()
        if not retval:
            break
        t += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff_gray = cv2.absdiff(gray, bkg_gray)
        diff_bgr = cv2.absdiff(frame, bkg_bgr)

        db, dg, dr = cv2.split(diff_bgr)
        ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY)
        ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY)
        ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY)
        # cv2.imshow("bb", bb)
        # cv2.imshow("bg", bg)
        # cv2.imshow("br", br)

        bImage = cv2.bitwise_or(bb, bg)
        bImage = cv2.bitwise_or(br, bImage)
        # cv2.imshow("b", bImage)

        # bImage = cv2.erode(bImage, None, 5)
        # bImage = cv2.dilate(bImage, None, 5)
        # bImage = cv2.erode(bImage, None, 7)

        contours, hierarchy = cv2.findContours(bImage, mode, method)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if AREA_TH < area < 1000:
                print(area)
                x, y, width, height = cv2.boundingRect(cnt)
                if width < 50:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

        cv2.imshow("frame", frame)
        cv2.imshow("bImage", bImage)
        cv2.imshow("diff_bgr", diff_bgr)

        if cv2.waitKey(1) == 27:
            break

    except KeyboardInterrupt:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()