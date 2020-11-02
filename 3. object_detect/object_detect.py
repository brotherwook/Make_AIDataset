import cv2
import numpy as np

#%%
background_path = "C:/MyWorkspace/Make_AIDataset/backgrounds/xxx0_background.png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('C:/MyWorkspace/Make_AIDataset/inputs/F20003;4_8sxxxx0.avi')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 20  # Binary Threshold
AREA_TH = 30  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

while True:
    try:
        retval, frame = cap.read()

        if not retval:
            break
        t += 1

        # roi_img = cv2.bitwise_and(frame, mask1)
        # cv2.imshow("roi_img", roi_img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # diff_gray = cv2.absdiff(gray, bkg_gray)
        diff_bgr = cv2.absdiff(frame, bkg_bgr)

        db, dg, dr = cv2.split(diff_bgr)
        ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow("bb", bb)
        # cv2.imshow("bg", bg)
        # cv2.imshow("br", br)

        bImage = cv2.bitwise_or(bb, bg)
        bImage = cv2.bitwise_or(br, bImage)

        bImage = cv2.medianBlur(bImage, 5)
        # cv2.imshow("b", bImage)

        # bImage = cv2.erode(bImage, None, 5)
        bImage = cv2.dilate(bImage, None, 10)
        # bImage = cv2.erode(bImage, None, 7)

        contours, hierarchy = cv2.findContours(bImage, mode, method)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            if 80 < w < 600 and 10 < h < 300:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


        frame = cv2.resize(frame,(int(width/2), int(height/2)))
        bImage = cv2.resize(bImage,(int(width/2), int(height/2)))
        diff_bgr = cv2.resize(diff_bgr,(int(width/2), int(height/2)))

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