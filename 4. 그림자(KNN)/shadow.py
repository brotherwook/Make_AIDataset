import numpy as np
import cv2

cap = cv2.VideoCapture('C:/MyWorkspace/Detectioncode/inputs/F18003_2_202009140900.mp4')

fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=120)

#%%
background_path = "C:/MyWorkspace/Detectioncode/temp/F18003_2_202009140900_bgr.png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('C:/MyWorkspace/Detectioncode/inputs/F18003_2_202009140900.mp4')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 0  # Binary Threshold
AREA_TH = 0  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

roi1 = np.array([[
    (296, 232),
    (543, 465),
    (693, 468),
    (378, 226),
]], dtype=np.int32)

# mask 생성, 마스크를 적용하여 ROI를 제외한 나머지 부분을 0(검은색)으로 만들기 위해
mask1 = np.zeros_like(bkg_bgr)  # 이미지와 같은 크기의 마스크 생성
mask2 = np.zeros_like(bkg_gray)
# 마스크 적용
# 마스크(원본과 같은 사이즈의 크기)에서 roi영역만 255로 채움
cv2.fillPoly(mask1, roi1, (255, 255, 255))
cv2.fillPoly(mask2, roi1, (255, 255, 255))


#%%
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    roi_img = cv2.bitwise_and(frame, mask1)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    gray_roi = cv2.bitwise_and(fgmask, mask2)

    cv2.imshow("roi_img", roi_img)
    cv2.imshow("gray_roi", gray_roi)


    ret, bImage = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow("b", bImage)
    contours, hierarchy = cv2.findContours(bImage, mode, method)
    cv2.drawContours(roi_img, contours, -1, (255, 0, 0), 1)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > AREA_TH:
            # print(area)
            x, y, width, height = cv2.boundingRect(cnt)
            if width*height<60:
                continue
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()