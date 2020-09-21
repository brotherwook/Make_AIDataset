import numpy as np
import cv2
import argparse

#%% 마우스 클릭을 위한 부분 (마스크 만들기)
dir_del = None
clicked_points = []
clone = None

def MouseLeftClick(event, x, y, flags, param):
    # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((y, x))

        # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[1], point[0]), 2, (0, 255, 255), thickness=-1)
        cv2.imshow("image", image)


def GetArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Enter the image files path")
    ap.add_argument("--sampling", default=1, help="Enter the sampling number.(default = 1)")
    args = vars(ap.parse_args())
    path = args['path']
    sampling = int(args['sampling'])
    return path, sampling

#%%
# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009140700.avi')
if (not cap.isOpened()):
    print('Error opening video')

#
fgbg = cv2.createBackgroundSubtractorKNN()

#%%
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 0  # Binary Threshold
AREA_TH = 50  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

flag = 0
mask = None
roi = [[]]
#%%
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    cv2.imshow("fgmask", fgmask)
    # 마스크 설정 부분
    # 마우스로 보기를 원하는 부분 클릭하고 n누르면 해당부분만 확인
    # 원하는 부분 클릭후  다음 프레임에서 또 다시 클릭하면 모두 확인가능
    # r 누르면 영상 재생
    if flag == 0:
        clone = frame.copy()
        cv2.imshow("image", gray)
        key = cv2.waitKey(0)

        if mask is None:
            mask = np.zeros_like(fgmask)
        
        #클릭한거 저장 및 마스크 생성
        if key == ord('n'):
            text_output = 'sample1'
            text_output += "," + str(len(clicked_points))
            roi = [[]]
            for points in clicked_points:
                text_output += "," + str(points[0]) + "," + str(points[1])
                print("(" + str(points[1]) + ", " + str(points[0]) + ')')
                roi[0].append((points[1], points[0]))
            if len(roi[0]) > 0:
                roi = np.array(roi)
                cv2.fillPoly(mask, roi, (255,255,255))
                clicked_points = []
        
        # 클릭한거 취소
        if key == ord('b'):
            if len(clicked_points) > 0:
                clicked_points.pop()
                image = clone.copy()
                for point in clicked_points:
                    cv2.circle(image, (point[1], point[0]), 2, (0, 255, 255), thickness=-1)
                cv2.imshow("image", image)
        
        if key == ord('r'):
            cv2.destroyAllWindows()
            flag = 1

    else:
        if len(roi[0]) > 0:
            # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
            cv2.imshow("mask", mask)
            roi_img = cv2.bitwise_and(fgmask, mask)
            ret, bImage = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            ret, bImage = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            
        cv2.imshow("b", bImage)
        contours, hierarchy = cv2.findContours(bImage, mode, method)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if AREA_TH < area < 1000:
                print(area)
                x, y, width, height = cv2.boundingRect(cnt)
                if width < 25:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)


        cv2.imshow("frame", frame)
        k = cv2.waitKey(0)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()