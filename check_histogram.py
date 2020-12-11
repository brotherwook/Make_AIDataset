import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% 마우스 클릭을 위한 부분 (마스크 만들기)
clone = None
click = False
x1,y1 = -1,-1
roi = []

def MouseLeftClick(event, x, y, flags, param):
    global click
    global x1, y1
    global roi
    global roi_img

    image = clone.copy()

    if event == cv2.EVENT_RBUTTONDOWN:
        # 마우스를 누른 상태
        roi = []
        click = True
        x1, y1 = x,y
        roi.append((x1,y1))
        # print("사각형의 왼쪽위 설정 : (" + str(x1) + ", " + str(y1) + ")")

    elif event == cv2.EVENT_MOUSEMOVE:
        # 마우스 이동
        # 마우스를 누른 상태 일경우
        if click == True:
            cv2.rectangle(image,(x1,y1),(x,y),(255,0,0), 2)
            # print("(" + str(x1) + ", " + str(y1) + "), (" + str(x) + ", " + str(y) + ")")
            cv2.imshow("image", image)

    elif event == cv2.EVENT_RBUTTONUP:
        # 마우스를 때면 상태 변경
        click = False
        cv2.rectangle(image,(x1,y1),(x,y),(255,0,0), 2)
        cv2.imshow("image", image)
        roi.append((x1, y))
        roi.append((x, y))
        roi.append((x, y1))

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("X:",x,"Y:",y)
        print(roi_img[x, y])


def number(num):
 return '%03d' % num


# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)
cv2.namedWindow("diff_bgr")
cv2.setMouseCallback("diff_bgr", MouseLeftClick)

video_path = 'C:\MyWorkspace\Make_AIDataset\inputs\F20010_620201103_160101.mkv'
img_name = video_path[-25:-4]

background_path = "C:/MyWorkspace/Make_AIDataset/backgrounds/" + img_name + ".png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)
bkg_bgr = cv2.resize(bkg_bgr, None, fx=0.5, fy=0.5)

cap = cv2.VideoCapture(video_path)

flag = 0
mask = None
reshapeROI = None

save_path = "D:/custom_haarcascade"
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
cnt = 102
k=0
roi_img = None

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # img = cv2.imread("C:/MyWorkspace/Make_AIDataset/backgrounds/test3_bgr.png")
    height, width, channels = img.shape
    clone = img.copy()

    while True:
        if len(roi) > 1:
            # cv2.rectangle(img, roi[0], roi[2], color=(0, 0, 255), thickness=1)
            mask = np.zeros_like(img)
            roi = np.array(roi)
            roi = roi[np.newaxis, ...]
            cv2.fillPoly(mask, roi, (255, 255, 255))
            reshapeROI = roi.reshape(-1)


        key = cv2.waitKey(k)

        # if key == 32:
        if reshapeROI is None:
            cv2.imshow("image", img)
            continue
        else:
            k=0
        roi_img = img[reshapeROI[1]:reshapeROI[3], reshapeROI[0]:reshapeROI[4]]
        bkg_roi = bkg_bgr[reshapeROI[1]:reshapeROI[3], reshapeROI[0]:reshapeROI[4]]

        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        bkg_roi = cv2.cvtColor(bkg_roi, cv2.COLOR_BGR2HSV)
        diff_bgr = cv2.absdiff(roi_img, bkg_roi)

        db, dg, dr = cv2.split(diff_bgr)
        ret, bb = cv2.threshold(db, 121, 255, cv2.THRESH_BINARY)
        ret, bg = cv2.threshold(dg, 50, 255, cv2.THRESH_BINARY)
        ret, br = cv2.threshold(dr, 70, 255, cv2.THRESH_TOZERO)
        bImage = cv2.bitwise_or(bb, bg)
        bImage = cv2.bitwise_or(br, bImage)

        kernel = np.ones((5, 5), np.uint8)
        bImage = cv2.morphologyEx(bImage,cv2.MORPH_CLOSE,kernel)
        bImage = cv2.GaussianBlur(bImage,(3,3),1)
        contours, hierarchy = cv2.findContours(bImage, mode, method)

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # 이 부분만 인식할 차량, 버스 등에 따라 수정해주면 됨
            temp = bImage[y:y + h, x:x + w]
            if 10 < w < 350 and 10 < h < 350:
                temp = cv2.morphologyEx(temp,cv2.MORPH_DILATE,kernel)
            else:
                mask = np.zeros_like(temp)
                temp = cv2.bitwise_and(temp,mask)

            bImage[y:y + h, x:x + w] = temp

        contours, hierarchy = cv2.findContours(bImage, mode, method)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(diff_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x+reshapeROI[0], y+reshapeROI[1]), (reshapeROI[0] + x + w, reshapeROI[1]+y + h), (0, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow("bImage",bImage)
        cv2.imshow("diff_bgr", diff_bgr)
        cv2.imshow("image", img)


        if key == ord("n"):
            break
        if key == 27:
            flag = 1
            break
        break

    if flag == 1:
        break

cv2.destroyAllWindows()

# roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
# diff_bgr = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2HSV)
#
# cv2.imshow("test",roi_img)
# cv2.imshow("diff_bgr", diff_bgr)
# cv2.imshow("thresh", thresh)
#
# cnt += 1
# clone = img.copy()
# roi = []
# hist1 = np.sum(roi_img[:,:]/roi_img.shape[0], axis=0)
# hist2 = np.sum(diff_bgr[:,:]/diff_bgr.shape[0], axis=0)
# plt.hist(hist1)
# plt.title("tophat")
# plt.show()
# plt.hist(hist2)
# plt.title("diff_hsv")
# plt.show()