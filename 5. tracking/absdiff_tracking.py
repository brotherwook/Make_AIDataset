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
        self.center = ((bbox[0] + (bbox[2]/2)), (bbox[1] + (bbox[3]/2)))
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
            ctx, cty = ((bbox[0] + (bbox[2] / 2)), (bbox[1] + (bbox[3] / 2)))
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

            for j in range(len(temp)):
                for i, v in enumerate(roi[0]):
                    if i < len(roi[0]) - 1:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

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
        self.center = ((bbox[0] + (bbox[2]/2)), (bbox[1] + (bbox[3]/2)))
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
            ctx, cty = ((bbox[0] + (bbox[2] / 2)), (bbox[1] + (bbox[3] / 2)))
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

            for j in range(len(temp)):
                for i, v in enumerate(roi[0]):
                    if i < len(roi[0]) - 1:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

            self.t.append(t)
            self.positions.append((bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]))
            # cv2.imshow(str(self.id), frame)
            return 1
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            cv2.destroyWindow(str(self.id))
            return -1

#%%
def compare(bbox, Cars):
    global reshapeROI

    ctx, cty = (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2)))

    if ctx < reshapeROI[0] or ctx > reshapeROI[4] or cty < reshapeROI[1] or cty > reshapeROI[5]:
        return -1

    for car in Cars.object:
        distance = math.sqrt(math.pow(ctx - int(car.center[0]), 2) + math.pow(cty - int(car.center[1]), 2))
        if distance <= 80:
            return -1

    return 1

#%%
click = None
x1,y1 = None, None
roi = []
clone = None

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

video_path = 'D:/video/20201008/F20003_4_202010080900.avi'
img_name = video_path[-25:-4]

background_path = "C:/MyWorkspace/Make_AIDataset/backgrounds/" + img_name + ".png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

temp = []
reshapeROI = None
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
cap = cv2.VideoCapture(video_path)
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
half_h, half_w = (int(height/2), int(width/2))

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
total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total)

bkg_bgr = cv2.resize(bkg_bgr, (half_w,half_h))
while True:

    retval, frameo = cap.read()

    if not retval:
        break
    frame = cv2.resize(frameo, (half_w,half_h))
    clone = frame.copy()

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
        if reshapeROI is None:
            reshapeROI = [0, 0, 0, height, width, height, width, 0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff_bgr = cv2.absdiff(frame, bkg_bgr)

        db, dg, dr = cv2.split(diff_bgr)
        ret, bb = cv2.threshold(db, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, bg = cv2.threshold(dg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, br = cv2.threshold(dr, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        bImage = cv2.bitwise_or(bb, bg)
        bImage = cv2.bitwise_or(br, bImage)

        bImage = cv2.medianBlur(bImage, 5)

        bImage = cv2.dilate(bImage, None, 10)

        contours, hierarchy = cv2.findContours(bImage, mode, method)

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            # 이 부분만 인식할 차량, 버스 등에 따라 수정해주면 됨
            if 90 < w < 350 and 40 < h < 90:
                bbox = (x, y, w, h)
                create = compare(bbox, Cars)
                if create == 1:
                    if w < 200 and h < 70:
                        car = Car(frame, bbox, t, id)
                    else:
                        car = Bus(frame, bbox, t, id)

                    Cars.object.append(car)
                    id += 1

                blob.append((x, y, x + w, y + h))

        for i, car in enumerate(Cars.object):
            k = car.update(frame, t)
            if k == -1:
                removed.append(i)

        Cars.remove()

        removed = []

        for box in blob:
            cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]),(0,255,0),1)
        blob = []

        if cv2.waitKey(1) == 27:
            break

        for j in range(len(temp)):
            for i, v in enumerate(roi[0]):
                if i < len(roi[0]) - 1:
                    frameo = cv2.line(frameo, tuple(temp[j][i]*2), tuple(temp[j][i + 1]*2), (0, 0, 255), 2)
                else:
                    frameo = cv2.line(frameo, tuple(temp[j][i]*2), tuple(temp[j][0]*2), (0, 0, 255), 2)


        # Display result
        cv2.imshow("Tracking", frame)
        print("\r" + "진행중 {:.2f}%".format((t / total) * 100), end='')

        t += 1
        out.write(frameo)


cap.release()
cv2.destroyAllWindows()
out.release()

# %% 데이터 확인

for i, person in enumerate(Cars.object):
    makeXML(person.id, Cars.object)

for i in range(id):

    track = root.find('track[@id="' + str(i) + '"]')

    if len(track) <= 20:
        root.remove(track)

apply_indent(root)
tree = ElementTree(root)
tree.write('trackingtest.xml')  # xml 파일로 보내기