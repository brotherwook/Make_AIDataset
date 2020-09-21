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
        preposition = self.last_position()
        if (preposition[0] - margin < center[0] < preposition[0] + margin) and (preposition[1] - margin < center[1] < preposition[1] + margin):
            self.positions.append(center)
            self.t.append(t)
            self.updateframe = t
            return 0

        return -1

    def last_position(self):
        return self.positions[-1]

    def last_position2(self):
        return self.positions[-2]


class PersonCounter:
    def __init__(self):
        self.people = []
        self.cntperson = 0

    def update_people(self, blobs, t, frame):
        global removed

        if len(self.people) < 1:
            for i, center in enumerate(blobs):
                person = Person(center, t, self.cntperson)
                self.people.append(person)
                self.cntperson += 1
                cv2.rectangle(frame,(center[0] - h_width, center[1] - h_height), (center[0]+h_width,center[1]+h_height),(255,0,0),3)
                cv2.putText(frame, ("person %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2, (127, 255, 255), 2)
        else:
            for i, center in enumerate(blobs):
                for j, person in enumerate(self.people):
                    k = person.update_position(center, t)
                    if k != -1:
                        cv2.rectangle(frame, (center[0] - h_width, center[1] - h_height), (center[0] + h_width, center[1] + h_height),
                                      (255, 0, 0), 3)
                        cv2.putText(frame, ("person %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (127, 255, 255), 2)
                        break

                if k == -1:
                    person = Person(center, t, self.cntperson)
                    self.people.append(person)
                    self.cntperson += 1
                    cv2.rectangle(frame, (center[0] - h_width, center[1] - h_height), (center[0] + h_width, center[1] + h_height),
                                  (255, 0, 0), 3)
                    cv2.putText(frame, ("person %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                (127, 255, 255), 2)


            for j, person in enumerate(self.people):
                if t - person.updateframe > 200:
                    removed.append(person.id)

            self.removePerson(removed)
            removed = []

            self.people[:] = [person for person in self.people
                                if t - person.updateframe < 200]

    def removePerson(self, removed):
        for i, v in enumerate(removed):
            makeXML(removed[i], self.people)




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

# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009140800.avi')
if (not cap.isOpened()):
    print('Error opening video')



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
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 10))
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

    # 중심점 구하기
    for i, contour in enumerate(contours):
        cv2.fillPoly(dil, contour, 255)
        area = cv2.contourArea(contour)
        # print(area)
        if 500 < area:
            x, y, width, height = cv2.boundingRect(contour)
            # if i == 1:
            #     mask = np.zeros(frame.shape[:2],dtype="uint8")
            #     cv2.drawContours(mask, [contour], -1, 255, -1)
            #     mask = cv2.erode(mask, None, iterations=2)
            #     masks = cv2.resize(mask, (720, 405))
            #     cv2.imshow("mask", masks)
            #     mean = cv2.mean(frame, mask=mask)[:3]
            #     test = cv2.bitwise_and(frame, frame, mask=mask)
            #     tests = cv2.resize(test, (720, 405))
            #     cv2.imshow("test", tests)
            #     mean = np.mean(mean)
            #     # print(mean)

            if width < 30:
                center = (int(x + width / 2), int(y + height / 2))
                cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
                blobs.append(center)

    dils = cv2.resize(dil, (720, 405))
    cv2.imshow("aa", dils)

    return dil

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

    for i,person in enumerate(people):
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

                    if i < (len(centroid)-1):
                        box.attrib["outside"] = '0'
                    else:
                        box.attrib["outside"] = '1'

                    box.attrib["occluded"] = '0'
                    box.attrib["keyframe"] = '1'
                    box.attrib["xtl"] = str(centroid[i][0] - h_width / 2)
                    box.attrib["ytl"] = str(centroid[i][1] - h_height / 2)
                    box.attrib["xbr"] = str(centroid[i][0] + h_width / 2)
                    box.attrib["ybr"] = str(centroid[i][1] + h_height / 2)
                    break


#%% 그 외 전역변수
fgbg = cv2.createBackgroundSubtractorKNN()

height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

# 현재 프레임 번호 변수
t = 0

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

# 사람의 박스 크기
h_width = 20
h_height = 50

blobs = []
margin = 20
person_counter = None
removed = []

flag = 0
mask = None
roi = [[]]


#%%
initXML()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    t += 1

    # 마스크 설정 부분
    # 마우스로 보기를 원하는 부분 클릭하고 n누르면 해당부분만 확인
    # 원하는 부분 클릭후  다음 프레임에서 또 다시 클릭하면 모두 확인가능
    # r 누르면 영상 재생
    if flag == 0:
        clone = frame.copy()
        cv2.imshow("image", frame)
        key = cv2.waitKey(0)

        if mask is None:
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(frame)

        # 클릭한거 저장 및 마스크 생성
        if key == ord('n'):
            roi = [[]]
            for points in clicked_points:
                # print("(" + str(points[1]) + ", " + str(points[0]) + ')')
                roi[0].append((points[1], points[0]))
            if len(roi[0]) > 0:
                roi = np.array(roi)
                cv2.fillPoly(mask, roi, (255, 255, 255))
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
            roi_img = cv2.bitwise_and(frame, mask)
            dil = imagepreprocessing(roi_img)

        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            dil = imagepreprocessing(frame)

        if person_counter is None:
            person_counter = PersonCounter()

        person_counter.update_people(blobs, t, frame)

        blobs = []

        frames = cv2.resize(frame, (1080, 720))
        cv2.imshow("frame", frames)


        k = cv2.waitKey(1)
        if k == 27:
            break

        if k == ord("n"):
            print(t)


cap.release()
cv2.destroyAllWindows()

#%% xml파일 생성

for i, person in enumerate(person_counter.people):
    makeXML(person.id, person_counter.people)

for i in range(person_counter.cntperson):
    track = root.find('track[@id="' + str(i) + '"]')
    if len(track) == 1:
        root.remove(track)

tree = ElementTree(root)
tree.write('annotations.xml')  #xml 파일로 보내기





