import cv2
import numpy as np

#%% 마우스 클릭을 위한 부분 (마스크 만들기)
clone = None
click = False
x1,y1 = -1,-1
roi = []

def MouseLeftClick(event, x, y, flags, param):
    global click
    global x1, y1
    global roi

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


def number(num):
 return '%03d' % num


# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

video_src = 'C:/MyWorkspace/Make_AIDataset/inputs/test3.avi'
# video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)

flag = 0
mask = None
reshapeROI = None

save_path = "D:/custom_haarcascade"

cnt = 102
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img = cv2.imread("C:/MyWorkspace/Make_AIDataset/backgrounds/test3_bgr.png")
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


        cv2.imshow("image", img)
        key = cv2.waitKey(0)

        if key == 32:
            if reshapeROI is None:
                continue

            roi_img = img[reshapeROI[1]:reshapeROI[3], reshapeROI[0]:reshapeROI[4]]
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("test",roi_img)
            cv2.imshow("tes2t",gray)
            cv2.imwrite(save_path+"/n/" + str(number(cnt)) + ".jpg", roi_img)
            cv2.imwrite(save_path+"/n_gray/" + str(number(cnt)) + ".jpg", gray)
            cnt += 1
            clone = img.copy()
            roi = []

        if key == ord("n"):
            break
        if key == 27:
            flag = 1
            break
    if flag == 1:
        break

cv2.destroyAllWindows()