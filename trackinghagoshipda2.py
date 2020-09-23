import numpy as np
import cv2
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%%
root = Element('annotations')
track = None

#%% 사람 클래스
class Person:
    def __init__(self, position, startframe, id):
        self.positions = [position]
        self.t = [startframe]
        self.startframe = startframe
        self.updateframe = startframe
        # self.mean = mean # 안쓰면 지울것
        self.id = id
        track = SubElement(root, 'track')
        track.attrib["id"] = str(id)  # 차량 id\n"
        track.attrib["label"] = 'person'
        track.attrib["source"] = 'manual'

    def update_position(self, center, t):
        self.positions.append(center)
        self.t.append(t)
        self.updateframe = t

    def last_position(self):
        return self.positions[-1]

    def last_position2(self):
        return self.positions[-2]


class PersonCounter:
    def __init__(self):
        self.people = []
        self.cntperson = 0

    def update_people(self, blobs, t):
       for i,person in enumerate(self.people):
           if i < len(blobs):
               person.update_position(blobs[i],t)


    def createPerson(self, center):
        global h_num
        person = Person(center, t, self.cntperson)
        self.people.append(person)
        h_num.append(self.cntperson)
        self.cntperson += 1


    def removePerson(self, removed):
        global h_num
        for i, v in enumerate(removed):
            makeXML(removed[i], self.people)
            self.people[:] = [person for person in self.people
                              if person.id != removed[i]]
            h_num[:] = [v for v in h_num
                        if v != removed[i]]


#%% 마우스 클릭을 위한 부분 (마스크 만들기)
dir_del = None
clicked_points = []
clone = None
color = np.random.randint(0, 255, (200, 3))
person_counter = None
cnt = 0
h_num = []
k = 0
def MouseLeftClick(event, x, y, flags, param):
    global image
    global preimage
    global person_counter
    global t
    global cnt


    # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = frame.copy()

        if t == 0:
            person_counter.createPerson((x, y))
            cv2.putText(image, "id :" + str(cnt), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cnt += 1
            for i, point in enumerate(clicked_points):
                cv2.circle(image, (point[0], point[1]), 2, color[i].tolist(), thickness=-1)
                cv2.rectangle(image, (point[0] -h_width, point[1] - h_height),
                              (point[0] + h_width, point[1] + h_height), color[i].tolist(), 3)
                cv2.putText(image, ("person %d" % person_counter.people[i].id), (point[0], point[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                            (127, 255, 255), 2)
            cv2.imshow("image", image)
        else:
            if cnt >= len(h_num):
                print("createPerson")
                person_counter.createPerson((x, y))

            cv2.putText(image, "id :" + str(h_num[cnt]+1), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2, cv2.LINE_AA)
            for i, point in enumerate(clicked_points):
                cv2.circle(image, (point[0], point[1]), 2, color[i].tolist(), thickness=-1)
                cv2.rectangle(image, (point[0] -h_width, point[1] - h_height),
                              (point[0] + h_width, point[1] + h_height), color[i].tolist(), 3)
                cv2.putText(image, ("person %d" % person_counter.people[i].id), (point[0], point[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                            (127, 255, 255), 1)
            cnt += 1
            cv2.imshow("image", image)

        preimage = image

# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Make_AIDataset\inputs\F18006_2\F18006_2_202009140900.avi')
if (not cap.isOpened()):
    print('Error opening video')

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
    global width
    global height

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

                    xtl = centroid[i][0] - h_width / 2
                    ytl = centroid[i][1] - h_height / 2
                    xbr = centroid[i][0] + h_width / 2
                    ybr = centroid[i][1] + h_height / 2

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

#%% 그 외 전역변수
fgbg = cv2.createBackgroundSubtractorKNN()

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# 현재 프레임 번호 변수
t = 0

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

# 사람의 박스 크기
h_width = 10
h_height = 26

blobs = []
margin = 20
removed = []

flag = 0
mask = None
roi = [[]]

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
image = None
preimage = None

#%%
initXML()
person_counter = PersonCounter()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    clone = frame.copy()

    if flag == 0:
        if t % 30 == 0 or t == total_frames:
            a=1
            while True:
                if t == 0:
                    cv2.putText(clone, "id :" + str(cnt), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cnt += 1
                else:
                    if cnt < len(h_num):
                        cv2.putText(clone, "id :" + str(h_num[cnt]), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(clone, "id :" + str(cnt), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                    # cnt += 1
                cv2.imshow("image", clone)
                key = cv2.waitKey(0)

                if key == ord('a'):
                    if t!=0:
                        person_counter.update_people(clicked_points, t)
                    clicked_points = []
                    break

                elif key == ord('s'):
                    try:
                        removed.append(h_num[cnt])
                        print("removoovoeoeo")
                        person_counter.removePerson(removed)
                        if cnt < len(h_num):
                            cv2.putText(clone, "id :" + str(h_num[cnt]), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(clone, "id :" + str(cnt+a), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                        (0, 255, 0), 2, cv2.LINE_AA)
                            a += 1
                        if cnt != 0:
                            clone = image.copy()
                    except:
                        pass
                elif key == ord('d'):
                    clicked_points = []
                    break
                elif key == ord('f'):
                    clicked_points.pop()
                    cnt -= 1
                    clone = frame.copy()
                    cv2.putText(image, "id :" + str(h_num[cnt] + 1), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    for i, point in enumerate(clicked_points):
                        cv2.circle(clone, (point[0], point[1]), 2, color[i].tolist(), thickness=-1)
                        cv2.rectangle(clone, (point[0] - h_width, point[1] - h_height),
                                      (point[0] + h_width, point[1] + h_height), color[i].tolist(), 3)
                        cv2.putText(clone, ("person %d" % person_counter.people[i].id), (point[0], point[1]),
                                    cv2.FONT_HERSHEY_PLAIN, 2,
                                    (127, 255, 255), 1)
                elif key == 27:
                    flag = 1
                    break

                else:
                    clone = image.copy()

            cnt = 0

    else:
        break

    if preimage is not None:
        # image = cv2.resize(image,(1080,720))
        preimage = cv2.resize(preimage,(1080,720))

        cv2.imshow("prev", preimage)
        # cv2.imshow("image", image)
    t += 1
cap.release()
cv2.destroyAllWindows()

#%% 데이터 확인
#
for i, person in enumerate(person_counter.people):
    makeXML(person.id, person_counter.people)
#
# for i in range(person_counter.cntperson):
#     track = root.find('track[@id="' + str(i) + '"]')
#     print(len(track))
#     if len(track) <= 5:
#         root.remove(track)
#
apply_indent(root)

tree = ElementTree(root)
tree.write('test.xml')  #xml 파일로 보내기



