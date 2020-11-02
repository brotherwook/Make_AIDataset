import numpy as np
import cv2
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%%
root = Element('annotations')
track = None

#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Make_AIDataset\inputs\F20003;4_8sxxxx0.avi')
# cap = cv2.VideoCapture('C:/MyWorkspace/Make_AIDataset/inputs/test3.avi')

if (not cap.isOpened()):
    print('Error opening video')

#%% 사람 클래스
class Person:
    def __init__(self, position, startframe, id, box):
        self.positions = [position]
        self.t = [startframe]
        self.startframe = startframe
        self.updateframe = startframe
        # self.mean = mean # 안쓰면 지울것
        self.id = id
        self.vector = []
        self.box = [box]
        track = SubElement(root, 'track')
        track.attrib["id"] = str(id)  # 차량 id\n"
        track.attrib["label"] = 'person'
        track.attrib["source"] = 'manual'

    def update_position(self, center, t, box):
        global reshapeROI
        global removed
        global clone
        preposition = self.last_position()
        prebox = self.last_box()
        # prevector = self.last_vector()
        vector = self.get_vector(preposition, center)

        predict_x, predict_y = self.predict()

        if (preposition[0] - margin < center[0] < preposition[0] + margin) and (preposition[1] - margin < center[1] < preposition[1] + margin):
            # if self.updateframe - self.startframe > 3:
            #     self.positions.append((predict_x,predict_y))
            #     self.t.append(t)
            #     self.updateframe = t
            #     self.box.append(box)
            #     self.vector.append(vector)
            #     cv2.putText(clone, ("car %d" % self.id), (predict_x, predict_y), cv2.FONT_HERSHEY_PLAIN, 1,
            #                 (127, 255, 255), 2)
            #     cv2.imshow("clone", clone)
            #
            # else:
            iou = cal_iou(box, prebox)
            # if iou > 0.8:
            self.positions.append(center)
            self.t.append(t)
            self.updateframe = t
            self.box.append(box)
            self.vector.append(vector)

            return 0

        return -1

    def first_position(self):
        return self.positions[0]

    def last_position(self):
        return self.positions[-1]

    def last_position2(self):
        return self.positions[-2:-1]

    def last_box(self):
        return self.box[-1]

    def last_vector(self):
        return self.vector[-1]

    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values decrease in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx ** 2 + dy ** 2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx / dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx / dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx / dy))
            else:
                angle = 180.0

        return distance, angle, dx, dy

    def predict(self):

        a = self.first_position()
        a = np.array(a).squeeze()
        b = self.last_position()
        b = np.array(b).squeeze()
        c = self.last_position2()
        c = np.array(c).squeeze()

        if self.startframe == self.updateframe:
            return 0, 0
        if b[0] == c[0] and b[1] == c[1]:
            return 0, 0

        # velocity_distance = math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(b[1]-a[1],2))
        # velocity = velocity_distance/(self.updateframe - self.startframe)

        velocity = math.sqrt(math.pow(b[0] - c[0], 2) + math.pow(c[1]-b[1],2))

        angle_distance = math.sqrt(math.pow(b[0] - c[0],2) + math.pow(c[1]-b[1],2))
        theta = math.acos((c[1]-b[1])/angle_distance)

        predict_x = int(b[0] - velocity * math.sin(theta))
        predict_y = int(velocity * math.cos(theta) + b[1])

        # predict_x = b[0] - abs((a[0] - b[0])/(self.updateframe - self.startframe))
        # predict_y = b[1] + abs((b[1]-a[1])/(self.updateframe - self.startframe))

        if predict_x == b[0] and predict_y == b[1]:
            return 0, 0

        return predict_x, predict_y

class PersonCounter:
    def __init__(self):
        self.people = []
        self.cntperson = 0

    def update_people(self, blobs, t, frame, boxes):
        global removed
        global reshapeROI

        if len(self.people) < 1:
            for center, box in zip(blobs, boxes):
                person = Person(center, t, self.cntperson, box)
                self.people.append(person)
                self.cntperson += 1
                # cv2.rectangle(frame,(center[0] - h_width, center[1] - h_height), (center[0]+h_width,center[1]+h_height),(255,0,0),3)
                cv2.putText(frame, ("car %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 255), 2)
        else:
            for center, box in zip(blobs, boxes):
                for j, person in enumerate(self.people):
                    k = person.update_position(center, t, box)
                    if k != -1:
                        # cv2.rectangle(frame, (center[0] - h_width, center[1] - h_height), (center[0] + h_width, center[1] + h_height),
                        #               (255, 0, 0), 3)
                        cv2.putText(frame, ("car %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (127, 255, 255), 2)
                        break

                if k == -1:
                    if box[0] >= reshapeROI[6] - 20 or box[1] <= reshapeROI[7] + 5:
                        person = Person(center, t, self.cntperson, box)
                        self.people.append(person)
                        self.cntperson += 1
                        # cv2.rectangle(frame, (center[0] - h_width, center[1] - h_height), (center[0] + h_width, center[1] + h_height),
                        #               (255, 0, 0), 3)
                        cv2.putText(frame, ("car %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (127, 255, 255), 2)

            for j, person in enumerate(self.people):
                lastbox = person.last_box()
                if lastbox[0] < reshapeROI[2]+20 or lastbox[1] > reshapeROI[3] - 20 or (t - person.updateframe > 200):
                    removed.append(person.id)

            self.removePerson(removed)
            removed = []

            self.people[:] = [person for person in self.people
                                if t - person.updateframe <= 50]

    def removePerson(self, removed):
        for i, v in enumerate(removed):
            makeXML(removed[i], self.people)

#%%
def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

#%% 마우스 클릭을 위한 부분 (마스크 만들기)
dir_del = None
clicked_points = []
clone = None
click = None

# def MouseLeftClick(event, x, y, flags, param):
#     # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicked_points.append((x, y))
#
#         # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
#         image = clone.copy()
#         for point in clicked_points:
#             cv2.circle(image, (point[0], point[1]), 2, (0, 255, 255), thickness=-1)
#         cv2.imshow("image", image)

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
        print("width=",x-x1,"height=",y-y1)

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("X:",x,"Y:",y)

# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)


#%% 영상 전처리
def imagepreprocessing(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    # 그림자제거
    ret, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    # 노이즈 제거
    medianblur = cv2.medianBlur(thresh, 5)
    # cv2.imshow("median", medianblur)

    # 팽창연산
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dil = cv2.dilate(medianblur, k)
    # cv2.imshow("dil", dil)

    # 윤곽선찾기
    contours, hierachy = cv2.findContours(dil, mode, method)

    # 윤곽선 안을 흰색으로 채우기
    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(dil, contour, 255)

    # 다시한번 팽창 연산
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 10))
    dil = cv2.dilate(dil, k)

    # 다시 윤곽선 찾기
    contours, hierachy = cv2.findContours(dil, mode, method)


    boxes = []
    # 중심점 구하기
    for i, contour in enumerate(contours):
        cv2.fillPoly(dil, contour, 255)
        area = cv2.contourArea(contour)
        # print(area)
        # if 1000 > area:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x,y,x+w,y+h))
        center = (int(x + w / 2), int(y + h / 2))
        cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
        blobs.append(center)

    # dils = cv2.resize(dil, (720, 480))
    cv2.imshow("aa", dil)

    return dil, boxes

#%% xml 만드는 부분
def initXML():
    global root
    root = Element('annotations')

    SubElement(root, 'version').text = '1.1'

    meta = SubElement(root, 'meta')
    task = SubElement(meta, 'task')
    SubElement(task, 'id').text = '12937'
    SubElement(task, 'name').text = 'F18006_2'
    SubElement(task, 'size').text = '10575'
    SubElement(task, 'mode').text = 'interpolation'
    SubElement(task, 'overlap').text = '5'
    SubElement(task, 'bugtracker').text
    SubElement(task, 'created').text = '2020-09-15 07:01:49.569020+01:00'
    SubElement(task, 'updated').text = '2020-09-16 08:37:29.588763+01:00'
    SubElement(task, 'start_frame').text = '0'
    SubElement(task, 'stop_frame').text = '10574'
    SubElement(task, 'frame_filter').text
    SubElement(task, 'z_order').text = 'False'

    sub = SubElement(task, 'labels')
    ssub = SubElement(sub, 'label')
    SubElement(ssub, 'name').text = 'person'
    SubElement(ssub, 'color').text = '#c06060'
    SubElement(ssub, 'attributes').text

    sub2 = SubElement(task, 'segments')
    ssub2 = SubElement(sub2, 'segment')
    SubElement(ssub2, 'id').text = '18705'
    SubElement(ssub2, 'start').text = '0'
    SubElement(ssub2, 'stop').text = '10574'
    SubElement(ssub2, 'url').text = 'http://cvat.org/?id=18705'


    sub3 = SubElement(task, 'owner')
    SubElement(sub3, 'username').text = 'brotherwook'
    SubElement(sub3, 'email').text = 'jahanda@naver.com'

    SubElement(task, 'assignee')

    sub4 =SubElement(task, 'original_size')
    SubElement(sub4, 'width').text = '1920'
    SubElement(sub4, 'height').text = '1080'

    SubElement(meta, 'dumped').text = '2020-09-16 08:37:30.049858+01:00'
    SubElement(meta, 'source').text = 'F18006_2_202009140900.avi'  # 동영상 파일 이름\

def makeXML(id, people):
    global root
    global track

    for i, person in enumerate(people):
        if person.id == id:

            track = root.find('track[@id="' + str(id) + '"]')
            centroid = person.positions
            framet = person.t
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

                    xtl = (person.box[i][0]) * 2
                    ytl = (person.box[i][1]) * 2
                    xbr = (person.box[i][0] + person.box[i][2]) * 2
                    ybr = (person.box[i][1] + person.box[i][3]) * 2

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
                    # box.attrib["xtl"] = str(centroid[i][0] - h_width / 2)
                    # box.attrib["ytl"] = str(centroid[i][1] - h_height / 2)
                    # box.attrib["xbr"] = str(centroid[i][0] + h_width / 2)
                    # box.attrib["ybr"] = str(centroid[i][1] + h_height / 2)

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

#%% 그 외 전역변수
fgbg = cv2.createBackgroundSubtractorKNN()

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# 현재 프레임 번호 변수
t = 0

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

blobs = []
margin = 10
person_counter = None
removed = []

flag = 0
mask = None
roi = [[]]
temp = []

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

reshapeROI = None
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
# out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (1920, 1080))

#%%
initXML()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    t += 1
    frame = cv2.resize(frame,(int(width/2),int(height/2)))
    # 마스크 설정 부분
    # 마우스로 보기를 원하는 부분 클릭하고 n누르면 해당부분만 확인
    # 원하는 부분 클릭후  다음 프레임에서 또 다시 클릭하면 모두 확인가능
    # r 누르면 영상 재생
    # if flag == 0:
    #     while True:
    #         clone = frame.copy()
    #         cv2.imshow("image", frame)
    #         key = cv2.waitKey(0)
    #
    #         if mask is None:
    #             # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             mask = np.zeros_like(frame)
    #
    #         # 클릭한거 저장 및 마스크 생성
    #         if key == ord('n'):
    #             roi = [[]]
    #             for points in clicked_points:
    #                 # print("(" + str(points[1]) + ", " + str(points[0]) + ')')
    #                 roi[0].append((points[0], points[1]))
    #             if len(roi[0]) > 0:
    #                 roi = np.array(roi)
    #                 cv2.fillPoly(mask, roi, (255, 255, 255))
    #                 clicked_points = []
    #                 temp.append(roi[0])
    #
    #         # 클릭한거 취소
    #         if key == ord('b'):
    #             if len(clicked_points) > 0:
    #                 clicked_points.pop()
    #                 image = clone.copy()
    #                 for point in clicked_points:
    #                     cv2.circle(image, (point[0], point[1]), 2, (0, 255, 255), thickness=-1)
    #                 cv2.imshow("image", image)
    #
    #         if key == ord('l'):
    #             if len(clicked_points) > 0:
    #                 for i in range(len(clicked_points) - 1):
    #                     cv2.line(mask, clicked_points[i], clicked_points[i + 1], 0, 2, cv2.LINE_AA)
    #                 clicked_points = []
    #
    #         if key == ord('r'):
    #             cv2.destroyAllWindows()
    #             flag = 1
    #             break
    clone = frame.copy()
    if flag == 0:
        while True:
            if len(roi) > 1:
                cv2.rectangle(frame, roi[0], roi[2], color=(0, 0, 255), thickness=1)
                mask = np.zeros_like(frame)
                roi = np.array(roi)
                roi = roi[np.newaxis, ...]
                cv2.fillPoly(mask, roi, (255, 255, 255))
                reshapeROI = roi.reshape(-1)
                temp.append(roi[0])

            cv2.imshow("image", frame)
            key = cv2.waitKey(0)

            if key == ord("r"):
                flag = 1
                break

    if flag == 1:
        if len(roi[0]) > 0:
            # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
            cv2.imshow("mask", mask)
            roi_img = cv2.bitwise_and(frame, mask)
            dil, boxes = imagepreprocessing(roi_img)

            for j in range(len(temp)):
                for i, v in enumerate(roi[0]):
                    if i < len(roi[0]) - 1:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)

        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            dil, boxes = imagepreprocessing(frame)

        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)


        if person_counter is None:
            person_counter = PersonCounter()

        person_counter.update_people(blobs, t, frame, boxes)

        blobs = []

        # 영상 추출용
        # out.write(frame)
        cv2.imshow("frame", frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

        if k == ord("n"):
            print(t)


cap.release()
# out.release()
cv2.destroyAllWindows()

#%% 데이터 확인

for i, person in enumerate(person_counter.people):
    makeXML(person.id, person_counter.people)

for i in range(person_counter.cntperson):
    track = root.find('track[@id="' + str(i) + '"]')
    # print(len(track))
    if len(track) <= 1:
        root.remove(track)

apply_indent(root)

tree = ElementTree(root)
tree.write('cartest.xml')  #xml 파일로 보내기



