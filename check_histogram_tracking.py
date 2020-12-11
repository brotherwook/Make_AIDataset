import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, frame, bbox, t, id, label):
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
        track.attrib["label"] = label
        track.attrib["source"] = 'manual'


    def update(self,img, t):
        global temp
        global reshapeROI
        # global frameo

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
            # p3 = (int(bbox[0])*2, int(bbox[1])*2)
            # p4 = (int(bbox[0] + bbox[2])*2, int(bbox[1] + bbox[3])*2)
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            # cv2.rectangle(frameo, p3, p4, (255, 0, 0), 2, 1)

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
    global reshapeROI

    ctx, cty = (int(bbox[0] + (bbox[2] / 2)), int(bbox[1] + (bbox[3] / 2)))

    if ctx < reshapeROI[0]+50 or ctx > reshapeROI[4] - 50 or cty < reshapeROI[1]+50 or cty > reshapeROI[5]-50:
        return -1

    for car in Cars.object:
        distance = math.sqrt(math.pow(ctx - int(car.center[0]), 2) + math.pow(cty - int(car.center[1]), 2))
        if distance <= 80:
            return -1

    return 1

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

#%%
def number(num):
 return '%03d' % num


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
# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)
cv2.namedWindow("diff_bgr")
cv2.setMouseCallback("diff_bgr", MouseLeftClick)

#%%
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

#%%
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
k = 0
roi_img = None

Cars = AllCar()
id = 0
t = 0
removed = []

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # img = cv2.imread("C:/MyWorkspace/Make_AIDataset/backgrounds/test3_bgr.png")
    height, width, channels = img.shape
    clone = img.copy()

    while True:
        if flag == 0:
            if len(roi) > 1:
                # cv2.rectangle(img, roi[0], roi[2], color=(0, 0, 255), thickness=1)
                mask = np.zeros_like(img)
                roi = np.array(roi)
                roi = roi[np.newaxis, ...]
                cv2.fillPoly(mask, roi, (255, 255, 255))
                reshapeROI = roi.reshape(-1)


            key = cv2.waitKey(0)

            if reshapeROI is None:
                cv2.imshow("image", img)
                continue
            else:
                if key == 32:
                    flag = 1
        if flag == 1:
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
                if 20 < w < 350 and 20 < h < 350:
                    temp = cv2.morphologyEx(temp,cv2.MORPH_DILATE,kernel)
                else:
                    mask = np.zeros_like(temp)
                    temp = cv2.bitwise_and(temp,mask)

                bImage[y:y + h, x:x + w] = temp

            contours, hierarchy = cv2.findContours(bImage, mode, method)
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(diff_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2, cv2.LINE_AA)
                # cv2.rectangle(img, (x+reshapeROI[0], y+reshapeROI[1]), (reshapeROI[0] + x + w, reshapeROI[1]+y + h), (0, 255, 255), 2, cv2.LINE_AA)

                bbox = (x+reshapeROI[0], y+reshapeROI[1], w, h)
                create = compare(bbox, Cars)
                if create == 1:
                    if w < 30 and h < 30:
                        label = "이륜차"
                    if w < 150 and h < 150:
                        label = '승용차'
                    else:
                        label = '버스'
                    car = Car(img, bbox, t, id, label)

                    Cars.object.append(car)
                    id += 1

            for i, car in enumerate(Cars.object):
                k = car.update(img, t)
                if k == -1:
                    removed.append(i)

            Cars.remove()

            removed = []



            cv2.imshow("bImage",bImage)
            cv2.imshow("diff_bgr", diff_bgr)
            cv2.imshow("image", img)
            key = cv2.waitKey(1)

            if key == ord("n"):
                break
            if key == 27:
                flag = 2
                break
        break

    t += 1

    if flag == 2:
        break

cv2.destroyAllWindows()

#%%
cap.release()

#%% 데이터 확인

for i, person in enumerate(Cars.object):
    makeXML(person.id, Cars.object)

for i in range(id):

    track = root.find('track[@id="' + str(i) + '"]')

    if len(track) <= 20:
        root.remove(track)

apply_indent(root)
tree = ElementTree(root)
tree.write('trackingtest.xml')  # xml 파일로 보내기