import cv2
import numpy as np

#%%
background_path = "C:/MyWorkspace/Detectioncode/temp/F18003_2_202009140900_bgr.png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18003_2\F18003_2_202009140900.mp4')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 20  # Binary Threshold
AREA_TH = 30  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

roi1 = np.array([[
    (277, 218),
    (557, 473),
    (698, 468),
    (364, 214)
]], dtype=np.int32)

# mask 생성, 마스크를 적용하여 ROI를 제외한 나머지 부분을 0(검은색)으로 만들기 위해
mask1 = np.zeros_like(bkg_bgr)  # 이미지와 같은 크기의 마스크 생성
mask2 = np.zeros_like(bkg_gray)
# 마스크 적용
# 마스크(원본과 같은 사이즈의 크기)에서 roi영역만 255로 채움
cv2.fillPoly(mask1, roi1, (255, 255, 255))
cv2.fillPoly(mask2, roi1, (255, 255, 255))

# 차선 0으로 만들기
cv2.line(mask1,(241 ,185 ),(565 ,479 ), (0,0,0),1)
cv2.line(mask1,(261 ,185 ),(591 ,479 ), (0,0,0),1)
cv2.line(mask1,(274 ,186 ),(615 ,478 ), (0,0,0),2)
cv2.line(mask1,(288 ,188 ),(638 ,479 ), (0,0,0),2)
cv2.line(mask1,(303 ,189 ),(661 ,478 ), (0,0,0),2)

cv2.imshow("mask", mask1)

bkg_bgr1 = cv2.bitwise_and(bkg_bgr, mask1)
bkg_gray1 = cv2.bitwise_and(bkg_gray, mask2)

cv2.imshow("bkg_bgr", bkg_bgr1)
while True:
    try:
        retval, frame = cap.read()
        if not retval:
            break
        t += 1
        roi_img = cv2.bitwise_and(frame, mask1)
        cv2.imshow("roi_img", roi_img)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

        diff_gray = cv2.absdiff(gray, bkg_gray1)
        diff_bgr = cv2.absdiff(roi_img, bkg_bgr1)

        db, dg, dr = cv2.split(diff_bgr)
        ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
        cv2.drawContours(roi_img, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if  area > AREA_TH and area < 250:
                print(area)
                x, y, width, height = cv2.boundingRect(cnt)
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