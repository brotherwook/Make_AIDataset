import numpy as np
import cv2
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%% object class
class AllObject:
    def __init__(self, blobs):
        self.all_object = []
        for i,v in enumerate(blobs):
            person = Person(v)
            self.all_object.append(person)

    def update(self, object):
        self.all_object.append(object)

class Person:
    def __init__(self, position):
        self.label = '사람'
        self.center = position
        self.width = h_width
        self.height = h_height


class Motorcycle:
    def __init__(self, position):
        self.label = '이륜차'
        self.center = position
        self.width = motor_width
        self.height = motor_height

class Bicycle:
    def __init__(self, position):
        self.label = '자전거'
        self.center = position
        self.width = bi_width
        self.height = bi_height


#%% 마우스 클릭을 위한 부분 (마스크 만들기)
dir_del = None
clicked_points = []
clone = None

def MouseLeftClick(event, x, y, flags, param):
    global flag
    # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[0], point[1]), 2, (0, 255, 255), thickness=4)
            cv2.rectangle(image, (point[0] - half_width, point[1] - half_height),
                          (point[0] + half_width, point[1] + half_height), (255, 255, 255), 2)
        cv2.imshow("image", image)

        if flag == 1:
            compare((x,y), blobs, label)

# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

#%% 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Make_AIDataset\inputs\F20001;3_3\F20001;3_3sxxxx0;양재1동 23;양재환승센터;(Ch 01)_[20200923]100000-[20200923]100600(20200923_100000).avi')
imagename = 'F18006_2_202009140900'

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
        x, y, width, height = cv2.boundingRect(contour)

        # cv2.putText(frame, "width:" + str(width) + " , height:" + str(height) + ", Area:" + str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        #             (225, 0, 255), 1, cv2.LINE_AA)
        # cv2.rectangle(frame, (x, y),
        #               (x + width, y + height), (255, 0, 255), 2)
        # print(area)
        if 400 < area:
            if 15 < width < 35 and 55 <= height:
                center = (int(x + width / 2), int(y + height / 2))
                # cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
                blobs.append(center)

    # dils = cv2.resize(dil, (720, 405))
    # cv2.imshow("aa", dils)

    return dil

#%% xml 만드는 부분
root = Element('annotations')

def initXML():
    global root
    root = Element('annotations')

    SubElement(root, 'version').text = '1.1'

    meta = SubElement(root, 'meta')
    task = SubElement(meta, 'task')
    SubElement(task, 'id').text = '12937'
    SubElement(task, 'name').text = 'image_F18006_2_202009140900'
    SubElement(task, 'size').text = '365'
    SubElement(task, 'mode').text = 'annotation'
    SubElement(task, 'overlap').text = '0'
    SubElement(task, 'bugtracker').text
    SubElement(task, 'created').text = '2020-09-15 07:01:49.569020+01:00'
    SubElement(task, 'updated').text = '2020-09-16 08:37:29.588763+01:00'
    SubElement(task, 'start_frame').text = '0'
    SubElement(task, 'stop_frame').text = '364'
    SubElement(task, 'frame_filter').text
    SubElement(task, 'z_order').text = 'False'

    sub = SubElement(task, 'labels')
    ssub = SubElement(sub, 'label')
    SubElement(ssub, 'name').text = '사람'
    SubElement(ssub, 'color').text = '#1dc799'
    SubElement(ssub, 'attributes').text

    sub2 = SubElement(task, 'segments')
    ssub2 = SubElement(sub2, 'segment')
    SubElement(ssub2, 'id').text = '28'
    SubElement(ssub2, 'start').text = '0'
    SubElement(ssub2, 'stop').text = '364'
    SubElement(ssub2, 'url').text = 'http://222.107.208.162:8088/?id=28'


    sub3 = SubElement(task, 'owner')
    SubElement(sub3, 'username').text = 'brotherwook'
    SubElement(sub3, 'email').text = 'jahanda@naver.com'

    SubElement(task, 'assignee')

    SubElement(meta, 'dumped').text = '2020-09-16 08:37:30.049858+01:00'

def makeXML(id, name, all_object):
    global root

    image = root.find('image[@id="' + str(id) + '"]')
    if image is None:
        image = SubElement(root, 'image')
        image.attrib["id"] = str(id)
        image.attrib["name"] = name
        image.attrib["width"] = str(width)
        image.attrib["height"] = str(height)

    for i, v in enumerate(all_object):
        box = SubElement(image, 'box')
        box.attrib["label"] = v.label
        box.attrib["occluded"] = '0'
        box.attrib["source"] = 'manual'

        xtl = v.center[0] - v.width / 2
        ytl = v.center[1] - v.height / 2
        xbr = v.center[0] + v.width / 2
        ybr = v.center[1] + v.height / 2

        if xtl < 0:
            xtl = 0
        if ytl < 0:
            ytl = 0
        if xbr > width:
            xbr = width
        if ybr > height:
            ybr = height

        box.attrib["xtl"] = str(xtl)
        box.attrib["ytl"] = str(ytl)
        box.attrib["xbr"] = str(xbr)
        box.attrib["ybr"] = str(ybr)

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
def compare(click, blobs, label):
    global create
    global removed

    for j, center in enumerate(blobs):
        if (click[0] - margin < center[0] < click[0] + margin) and (click[1] - margin < center[1] < click[1] + margin):
            removed.append(center)
            removed.append(click)

    if click not in removed:
        create.append(click)

    for i, v in enumerate(create):
        blobs.append(v)
        if label.__eq__('사람'):
            obj = Person(v)
            everything.update(obj)
        elif label.__eq__('이륜차'):
            obj = Motorcycle(v)
            everything.update(obj)
        elif label.__eq__('자전거'):
            obj = Bicycle(v)
            everything.update(obj)

    # clicks[:] = clicks[1:]
    create = []

def removeBOX():
    global removed
    global blobs
    global everything

    blobs[:] = [v for j, v in enumerate(blobs)
                if v not in removed]

    everything.all_object[:] = [v for j, v in enumerate(everything.all_object)
                                if v.center not in removed]

    removed = []

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
create = []
id = 0

flag = 0
mask = None
roi = [[]]
temp = []

color = np.random.randint(0, 255, (200, 3))

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 동영상 저장용 (안씀)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (1920, 1080))


# 사람의 박스 크기
h_width = 20
h_height = 50

# 이륜차
motor_width = 60
motor_height = 60

# 자전거
bi_width = 40
bi_height = 50

# label 종류에 따른 변수
label = '사람'
label_english = 'person'
label_width = 20
label_height = 50

half_width = int(label_width / 2)
half_height = int(label_height / 2)

everything = None

#%%
initXML()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(int(width/2),int(height/2)))

    # 마스크 설정 부분
    # 마우스로 보기를 원하는 부분 클릭하고 n누르면 해당부분만 확인
    # 원하는 부분 클릭후  다음 프레임에서 또 다시 클릭하면 모두 확인가능
    # r 누르면 영상 재생
    if flag == 0:
        while True:
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
                    roi[0].append((points[0], points[1]))
                if len(roi[0]) > 0:
                    roi = np.array(roi)
                    cv2.fillPoly(mask, roi, (255, 255, 255))
                    clicked_points = []
                    temp.append(roi[0])

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
                clicked_points = []
                flag = 1
                break


    if flag == 1:
        blobs = []

        # 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", MouseLeftClick)
        name = imagename + "_" + str(t) +".jpg"
        if len(roi[0]) > 0:
            # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
            cv2.imshow("mask", mask)
            roi_img = cv2.bitwise_and(frame, mask)
            dil = imagepreprocessing(roi_img)

            for j in range(len(temp)):
                for i,v in enumerate(roi[0]):
                    if i < len(roi[0])-1:
                        frame = cv2.line(frame,tuple(temp[j][i]),tuple(temp[j][i+1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)
        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            dil = imagepreprocessing(frame)

        if t % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
            everything = AllObject(blobs)
            while True:
                # cv2.imwrite('C:\MyWorkspace\Make_AIDataset\image_F18005_1_202009140700/original/' + name, frame)
                clone = frame.copy()
                cv2.putText(clone, label_english, (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                half_width = int(label_width / 2)
                half_height = int(label_height / 2)

                for i,obj in enumerate(everything.all_object):
                    cv2.rectangle(clone,(obj.center[0] - int(obj.width/2), obj.center[1] - int(obj.height/2)),
                                  (obj.center[0]+int(obj.width/2), obj.center[1]+int(obj.height/2)), (255, 255, 255), 2)
                    cv2.circle(clone, (obj.center[0], obj.center[1]), 2, (255, 0, 0), thickness=2)


                # frames = cv2.resize(frame, (1080, 720))
                cv2.imshow("image", clone)
                cv2.imshow("dil", dil)
                k = cv2.waitKey(0)

                # 프로그램 종료
                if k == 27:
                    flag = 2
                    break

                # 다음 프레임으로 넘어가기
                if k == ord("d"):
                    makeXML(id, name, everything.all_object)
                    id += 1
                    clicked_points = []
                    blobs = []
                    # cv2.imwrite('C:\MyWorkspace\Make_AIDataset\image_F18005_1_202009140700/detect/' + name, clone)

                    break

                if k == ord('s'):
                    # compare(clicked_points, blobs)
                    removeBOX()
                    clicked_points = []

                # 사람 1번
                if k == 49:
                    label = '사람'
                    label_english = 'person'
                    label_width = h_width
                    label_height = h_height

                # 이륜차 2번
                if k == 50:
                    label = '이륜차'
                    label_english = 'motorcycle'
                    label_width = motor_width
                    label_height = motor_height

                # 자전거 3번
                if k == 51:
                    label = '자전거'
                    label_english = 'bicycle'
                    label_width = bi_width
                    label_height = bi_height
    else:
        break

    t += 1


cap.release()
out.release()
cv2.destroyAllWindows()

#%% 데이터 확인

apply_indent(root)

tree = ElementTree(root)
tree.write('ss.xml')  #xml 파일로 보내기



