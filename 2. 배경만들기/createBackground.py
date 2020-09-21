import cv2
import numpy as np

img_name = 'F18006_2_202009112100'

# 비디오 불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009112100.avi')
if (not cap.isOpened()):
    print('Error opening video')

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

acc_gray = np.zeros(shape=(height, width), dtype=np.float32)
acc_bgr = np.zeros(shape=(height, width, 3), dtype=np.float32)
t = 0

while True:
    try:
        retval, frame = cap.read()
        if not retval:
            break
        t += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.accumulate(gray, acc_gray)
        avg_gray = acc_gray / t
        dst_gray = cv2.convertScaleAbs(avg_gray)

        cv2.accumulate(frame, acc_bgr)
        avg_bgr = acc_bgr / t
        dst_bgr = cv2.convertScaleAbs(avg_bgr)

        cv2.imshow("frame", frame)
        cv2.imshow("acc_gray", dst_gray)
        cv2.imshow("acc_bgr", dst_bgr)

        if cv2.waitKey(1) == 27:
            break

    except KeyboardInterrupt:
        break



if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()
cap.release()

# 마지막 이미지 저장
cv2.imwrite('C:/MyWorkspace/Detectioncode/temp/' + img_name + '_gray.png', dst_gray)
cv2.imwrite('C:/MyWorkspace/Detectioncode/temp/' + img_name + '_bgr.png', dst_bgr)
