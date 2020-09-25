import cv2
import numpy as np

trackers = [cv2.TrackerBoosting_create,
           cv2.TrackerMIL_create,
           cv2.TrackerKCF_create,
           cv2.TrackerTLD_create,
           cv2.TrackerMedianFlow_create,
           cv2.TrackerGOTURN_create,
           cv2.TrackerCSRT_create,
           cv2.TrackerMOSSE_create]

trackerIdx = 0
tracker = []
isFirst = True
roi = []
remove = []
bbox = []
trackerName = 'TrackerBoosting'

video_src = 'C:\MyWorkspace\Make_AIDataset\inputs\F18006_2\F18006_2_202009140900.avi'
cap = cv2.VideoCapture(video_src)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
win_name = 'Tracking APIs'
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_draw = frame.copy()
    if not(len(tracker) > 0):
        cv2.putText(img_draw, "Press the Space to set ROI!!", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2,cv2.LINE_AA)
    else:
        print(len(tracker))
        for i in range(len(tracker)):
            ok, temp = tracker[i].update(frame)
            if ok:
                bbox.append(temp)
            else:
                print("remove", i)
                remove.append(i)

        for i, box in enumerate(bbox):
            (x,y,w,h) = box
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 1)
        bbox = []
        if len(remove)>1:
            for i in range(len(remove)-1, 0, -1):
                del tracker[i]


        remove = []
    cv2.putText(img_draw, str(trackerIdx) + ":" + str(trackerName), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(win_name, img_draw)
    key = cv2.waitKey(1)

    if key == ord(' ') or (video_src != 0 and isFirst):
        isFirst = False
        roi = (cv2.selectROIs(win_name, frame, False))
        if len(roi)> 0:
            if roi[0, 2] and roi[0, 3]:
                for i in range(len(roi)):
                    tracker.append(trackers[trackerIdx]())
                    isInint = tracker[i].init(frame, tuple(roi[i]))
                trackerName = tracker[-1].__class__.__name__

    elif key in range(48, 56):
        trackerIdx = key-48
        if bbox is not None:
            for i in range(len(roi)):
                tracker.append(trackers[trackerIdx]())
                isInint = tracker[i].init(frame, tuple(roi[i]))
            trackerName = tracker[-1].__class__.__name__

    elif key == 27:
        break

    else:
        pass

cap.release()
cv2.destroyAllWindows()

