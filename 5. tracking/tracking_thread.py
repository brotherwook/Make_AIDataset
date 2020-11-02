import cv2
import sys
import threading
import numpy as np

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


class Car(threading.Thread):
    def __init__(self, frame, bbox, cap, t, id, tracker):
        threading.Thread.__init__(self,name=id)
        self.bbox = bbox
        self.cap = cap
        self.height, self.width = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.t = t
        self.id = id
        self.tracker = tracker
        ok = self.tracker.init(frame, bbox)


    def run(self):
        t = 0
        print(self.id,"thread start")
        while True:
            ret, img = self.cap.read()
            img = cv2.resize(img, (int(self.width / 2), int(self.height / 2)))

            if t < self.t:
                t += 1
            else:
                # Update tracker
                ok, bbox = self.tracker.update(img)

                # Draw bounding box
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)
                else:
                    # Tracking failure
                    cv2.putText(img, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)
                    break

                cv2.imshow(str(self.id), img)
                key = cv2.waitKey(33)
                if key == 27:
                    break

        self.cap.release()
        cv2.destroyWindow(str(self.id))

click = None
x1,y1 = None, None
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
            cv2.imshow("Tracking", image)

    elif event == cv2.EVENT_RBUTTONUP:
        # 마우스를 때면 상태 변경
        click = False
        cv2.rectangle(image,(x1,y1),(x,y),(255,0,0), 2)
        cv2.imshow("Tracking", image)
        roi.append((x1, y))
        roi.append((x, y))
        roi.append((x, y1))
        print("width=",x-x1,"height=",y-y1)

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("X:",x,"Y:",y)



frame = None

if __name__ == '__main__':
    # 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", MouseLeftClick)

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['KCF','MEDIANFLOW', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[1]
    # KCF, MEDIANFLOW
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    cap = cv2.VideoCapture("C:/MyWorkspace/Make_AIDataset/inputs/F20003;4_8sxxxx0.avi")
    height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    # Exit if video not opened.
    if not cap.isOpened():
        print("Could not open video")
        sys.exit()

    fgbg = cv2.createBackgroundSubtractorKNN()
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE

    # Read first frame.
    ret, frame = cap.read()
    if not ret:
        print('Cannot read video file')
        sys.exit()
    # frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
    # for i in range(10):
    #     _, frame = cap.read()
    #     fgmask = fgbg.apply(frame)

    cap = cv2.VideoCapture("C:/MyWorkspace/Make_AIDataset/inputs/F20003;4_8sxxxx0.avi")
    t = 0
    id = 0
    blob = []
    flag = 0
    roi = []
    reshapeROI = None
    car =None
    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (int(width/2),int(height/2)))

        if flag == 0:
            flag = 1
            clone = frame.copy()
            while True:
                if len(roi) > 1:
                    cv2.rectangle(frame, roi[0], roi[2], color=(0, 0, 255), thickness=1)
                    mask = np.zeros_like(frame)
                    roi = np.array(roi)
                    roi = roi[np.newaxis, ...]
                    cv2.fillPoly(mask, roi, (255, 255, 255))
                    reshapeROI = roi.reshape(-1)

                cv2.imshow("Tracking", frame)
                key = cv2.waitKey(0)

                if key == ord("r"):
                    flag = 1
                    break

        if flag == 1:

            # roi_img = cv2.bitwise_and(frame, mask)
            fgmask = fgbg.apply(frame)

            if t < 4:
                t += 1
                continue
            # Start timer
            timer = cv2.getTickCount()

            ret, bImage = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(bImage, mode, method)

            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 90 and h > 40:
                    bbox = (x, y, w, h)
                    if reshapeROI[0]-5 <= x <= reshapeROI[0]+5 or reshapeROI[1]-5 <= y <= reshapeROI[1]+5 or reshapeROI[4]-5 <= x <= reshapeROI[4]+5 or reshapeROI[5]-5 <= y <= reshapeROI[5]+10:
                        if car is None:
                            car = Car(frame,bbox,cap,t, id,tracker)
                            car.start()
                            id += 1
                    blob.append((x, y, x+w, y+h))

            for box in blob:
                cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]),(0,255,0),1)
            blob = []
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            # Display result
            cv2.imshow("Tracking", frame)
            t += 1
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()