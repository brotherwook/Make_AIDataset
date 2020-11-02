import cv2
import sys
import numpy as np
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%%
root = Element('annotations')
track = None

#%%
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

#%%
class AllCar():
    def __init__(self):
        self.object = []

    def remove(self):
        global removed

        for i, v in enumerate(removed):
            makeXML(self.object[removed[i]].id, self.object)

        self.object[:] = [car for i, car in enumerate(self.object)
                          if i not in removed]

class Car():
    def __init__(self, frame, bbox, t, id):
        self.bbox = bbox
        self.positions = [(bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])]
        self.t = [t]
        self.id = id
        self.tracker = cv2.TrackerMedianFlow_create()
        self.center = (int(bbox[0] + (bbox[2]/2)), int(bbox[1] + (bbox[3]/2)))
        ok = self.tracker.init(frame, bbox)
        self.cnt = 0
        track = SubElement(root, 'track')
        track.attrib["id"] = str(id)  # 차량 id\n"
        track.attrib["label"] = '승용차'
        track.attrib["source"] = 'manual'


    def update(self,img, t):
        global temp
        global reshapeROI

        frame = img.copy()
        # Update tracker
        ok, bbox = self.tracker.update(frame)
        precenter = self.center
        # Draw bounding box
        if ok:
            ctx, cty = (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2)))
            self.center = (ctx, cty)
            if precenter[0] == ctx and precenter[1] == cty:
                self.cnt += 1
            else:
                self.cnt = 0

            if self.cnt > 20:
                cv2.destroyWindow(str(self.id))
                return -1

            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            if ctx < reshapeROI[0] or ctx > reshapeROI[4] or cty < reshapeROI[1] or cty > reshapeROI[5]:
                cv2.destroyWindow(str(self.id))
                return -1

            # for j in range(len(temp)):
            #     for i, v in enumerate(roi[0]):
            #         if i < len(roi[0]) - 1:
            #             frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
            #         else:
            #             frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

            self.t.append(t)
            self.positions.append((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
            cv2.imshow(str(self.id), frame)
            return 1
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            cv2.destroyWindow(str(self.id))
            return -1

class Bus():
    def __init__(self, frame, bbox, t, id):
        self.bbox = bbox
        self.positions = [(bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3])]
        self.t = [t]
        self.id = id
        self.tracker = cv2.TrackerMedianFlow_create()
        self.center = (int(bbox[0] + (bbox[2]/2)), int(bbox[1] + (bbox[3]/2)))
        ok = self.tracker.init(frame, bbox)
        self.cnt = 0
        track = SubElement(root, 'track')
        track.attrib["id"] = str(id)  # 차량 id\n"
        track.attrib["label"] = '버스'
        track.attrib["source"] = 'manual'


    def update(self,img, t):
        global temp
        global reshapeROI

        frame = img.copy()
        # Update tracker
        ok, bbox = self.tracker.update(frame)
        precenter = self.center
        # Draw bounding box
        if ok:
            ctx, cty = (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2)))
            self.center = (ctx, cty)
            if precenter[0] == ctx and precenter[1] == cty:
                self.cnt += 1
            else:
                self.cnt = 0

            if self.cnt > 20:
                cv2.destroyWindow(str(self.id))
                return -1

            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            if ctx < reshapeROI[0] or ctx > reshapeROI[4] or cty < reshapeROI[1] or cty > reshapeROI[5]:
                cv2.destroyWindow(str(self.id))
                return -1

            # for j in range(len(temp)):
            #     for i, v in enumerate(roi[0]):
            #         if i < len(roi[0]) - 1:
            #             frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
            #         else:
            #             frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

            self.t.append(t)
            self.positions.append((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
            cv2.imshow(str(self.id), frame)
            return 1
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            cv2.destroyWindow(str(self.id))
            return -1

#%%
def compare(bbox, Cars):
    ctx, cty = (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2)))
    for car in Cars.object:
        distance = math.sqrt(math.pow(ctx - car.center[0], 2) + math.pow(cty - car.center[1], 2))
        if distance <= 80:
            return -1

    return 1

#%%
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

#%%
def makeXML(id, Cars):
    global root
    global track
    global width
    global height

    for car in Cars:
        if car.id == id:

            track = root.find('track[@id="' + str(id) + '"]')
            centroid = car.positions
            framet = car.t
            if track is None:
                print("nononononon")
            else:
                for i in range(len(centroid)):
                    box = SubElement(track, 'box')
                    box.attrib["frame"] = str(framet[i])  # 박스의 프레임 넘버

                    if i < (len(centroid) - 1):
                        box.attrib["outside"] = '0'
                    else:
                        box.attrib["outside"] = '1'

                    xtl = (centroid[i][0]) * 2
                    ytl = (centroid[i][1]) * 2
                    xbr = (centroid[i][2]) * 2
                    ybr = (centroid[i][3]) * 2

                    if xtl < 0:
                        xtl = 0
                    if ytl < 0:
                        ytl = 0
                    if xbr > width:
                        xbr = width
                    if ybr > height:
                        ybr = height

                    box.attrib["occluded"] = '0'
                    box.attrib["keyframe"] = '1'
                    box.attrib["xtl"] = str(xtl)
                    box.attrib["ytl"] = str(ytl)
                    box.attrib["xbr"] = str(xbr)
                    box.attrib["ybr"] = str(ybr)

            break

def apply_indent(elem, level = 0):
    # tab = space * 2
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for elem in elem:
            apply_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent



#%%
temp = []
reshapeROI = None
width = None
height = None

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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
    out = cv2.VideoWriter('C:/MyWorkspace/Make_AIDataset/outputs/video/output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (width, height))

    t = 0
    id = 0
    blob = []
    flag = 0
    roi = []
    car =None
    Cars = AllCar()
    removed = []
    while True:
        # Read a new frame
        ret, frameo = cap.read()
        if not ret:
            break

        frame = cv2.resize(frameo, (int(width/2),int(height/2)))

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
                    temp.append(roi[0])

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

            ret, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

            # 노이즈 제거
            medianblur = cv2.medianBlur(thresh, 5)
            # cv2.imshow("median", medianblur)

            # 팽창연산
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
            dil = cv2.dilate(medianblur, k)
            # cv2.imshow("dil", dil)

            cv2.imshow("fgmask", dil)

            # 윤곽선찾기
            contours, hierachy = cv2.findContours(dil, mode, method)

            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if 90 < w < 350 and 40 < h < 150:
                    bbox = (x, y, w, h)
                    # if reshapeROI[0]-5 <= x <= reshapeROI[0]+5 or reshapeROI[1]-5 <= y <= reshapeROI[1]+5 or reshapeROI[4]-5 <= x <= reshapeROI[4]+5 or reshapeROI[5]-5 <= y <= reshapeROI[5]+10:
                    # if x <= reshapeROI[0]+5 or y <= reshapeROI[1] + 5 or x >= reshapeROI[4] - 5 or y >= reshapeROI[5]-5:
                    create = compare(bbox, Cars)
                    if create == 1:
                        if w < 200 and h < 70:
                            car = Car(frame, bbox, t, id)
                        else:
                            car = Bus(frame, bbox, t, id)

                        Cars.object.append(car)
                        id += 1

                    blob.append((x, y, x+w, y+h))

            for i, car in enumerate(Cars.object):
                k = car.update(frame,t)
                if k == -1:
                    removed.append(i)

            Cars.remove()

            removed = []

            # for box in blob:
            #     cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]),(0,255,0),1)
            blob = []
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # # Display tracker type on frame
            # cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
            #
            # # Display FPS on frame
            # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            for j in range(len(temp)):
                for i, v in enumerate(roi[0]):
                    if i < len(roi[0]) - 1:
                        frameo = cv2.line(frameo, tuple(temp[j][i]*2), tuple(temp[j][i + 1]*2), (0, 0, 255), 2)
                    else:
                        frameo = cv2.line(frameo, tuple(temp[j][i]*2), tuple(temp[j][0]*2), (0, 0, 255), 2)
            # Display result
            cv2.imshow("Tracking", frame)
            t += 1

            out.write(frameo)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    out.release()

    # %% 데이터 확인

    for i, person in enumerate(Cars.object):
        makeXML(person.id, Cars.object)

    for i in range(id):
        track = root.find('track[@id="' + str(i) + '"]')
        # print(len(track))
        if len(track) <= 20:
            root.remove(track)

    apply_indent(root)

    tree = ElementTree(root)
    tree.write('trackingtest.xml')  # xml 파일로 보내기
