import numpy as np
import cv2
import tensorflow_hub as hub
from xml.etree.ElementTree import Element, SubElement, ElementTree

#%%
root = Element('annotations')
track = None

#%% 마우스 클릭을 위한 부분 (마스크 만들기)
dir_del = None
clicked_points = []
clone = None

def MouseLeftClick(event, x, y, flags, param):
    # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[0], point[1]), 2, (0, 255, 255), thickness=4)
            cv2.rectangle(image, (point[0] - half_width, point[1] - half_height), (point[0] + half_width, point[1] + half_height), (255, 255, 255), 3)
        cv2.imshow("image", image)

# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

#%% 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Make_AIDataset\inputs\F20001;3_3sxxxx0;양재1동 23;양재환승센터;(Ch 01)_[20200928]080000-[20200928]080600(20200928_080000).avi')
imagename = 'F20001;3_3'

if (not cap.isOpened()):
    print('Error opening video')

#%% tensorhub 사용
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1")

def deeplearning(output, clone):
    for i, score in enumerate(output["detection_scores"][0]):
        if score < threshold:
            break
        ymin, xmin, ymax, xmax = output["detection_boxes"][0][i]
        (left, top) = (int(xmin * width), int(ymin * height))
        (right, bottom) = (int(xmax * width), int(ymax * height))
        if (right - left) > 30:  # 가로가 너무 크거나
            continue
        if (bottom - top) > 70:  # 세로가 너무 클 경우 무시
            continue
        if (right - left) * (bottom - top) < 800:  # 너무 작은것도 무시
            continue
        if output["detection_classes"][0][i] == 1:  # <= 8: # 1~8 = 사람,자전거,승용차,오토바이,비행기,버스,전철,트럭
            # 참고: https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt
            center = ((xmin + xmax) * width / 2, (ymin + ymax) * height / 2)
            blobs.append(center)
            class_entity = "{:.2f}%".format(score * 100)
            cv2.putText(clone, class_entity, (left, bottom + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                        (255, 255, 255), 1)

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

def makeXML(id, name, blobs):
    global root

    image = SubElement(root, 'image')
    image.attrib["id"] = str(id)  # 차량 id\n"
    image.attrib["name"] = name
    image.attrib["width"] = str(width)
    image.attrib["height"] = str(height)

    for i in range(len(blobs)):
        box = SubElement(image, 'box')
        box.attrib["label"] = '사람'  # 박스의 프레임 넘버
        box.attrib["occluded"] = '0'
        box.attrib["source"] = 'manual'

        xtl = blobs[i][0] - h_width / 2
        ytl = blobs[i][1] - h_height / 2
        xbr = blobs[i][0] + h_width / 2
        ybr = blobs[i][1] + h_height / 2

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


#%% 마우스 클릭과 비교해서 삭제할지 생성할지 정하는 함수
def compare(clicks, blobs):
    global create
    global removed

    # 클릭한 값들 중 blobs안에 있는 값과 가까이 있는 값들 removed에 추가
    for i, click in enumerate(clicks):
        for j, center in enumerate(blobs):
            if (click[0] - margin < center[0] < click[0] + margin) and (click[1] - margin < center[1] < click[1] + margin):
                removed.append(center)
                removed.append(click)

    # blobs 안에 있는 값들 중 removed 안에 포함되지 않은 것만 blobs에 저장
    blobs[:] = [v for j,v in enumerate(blobs)
                if v not in removed]

    # clicks 안에 있는 값들 중 removed 안에 포함되지 않은 것만 create에 저장
    create[:] = [v for i, v in enumerate(clicks)
                 if v not in removed]

    # create 안에 있는 값들 blobs에 추가
    for i, v in enumerate(create):
        blobs.append(v)

    # 초기화
    removed = []
    create = []

#%% 그 외 전역변수
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
# height, width = (640, 860)

# 현재 프레임 번호 변수
t = 0

# 사람의 박스 크기
h_width = 20
h_height = 50
half_width = int(h_width / 2)
half_height = int(h_height / 2)

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
out = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

# tfhub용
threshold = 0.2

#%%
initXML()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(frame,(860,640))

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
            input = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            input = input[np.newaxis, ...]

            for j in range(len(temp)):
                for i,v in enumerate(roi[0]):
                    if i < len(roi[0])-1:
                        frame = cv2.line(frame,tuple(temp[j][i]),tuple(temp[j][i+1]), (0, 0, 255), 2)
                    else:
                        frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)
        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input = input[np.newaxis, ...]

        if t % int(cap.get(cv2.CAP_PROP_FPS)) == 0:
            output = model(input)
            deeplearning(output, clone)
            while True:
                # cv2.imwrite('C:\MyWorkspace\Make_AIDataset\image_F18005_1_202009140700/original/' + name, frame)
                clone = frame.copy()
                for i,center in enumerate(blobs):
                    cv2.rectangle(clone,(center[0] - half_width , center[1] - half_height), (center[0]+half_width, center[1]+half_height), (255, 255, 255), 2)
                    cv2.circle(clone, (center[0], center[1]), 2, (255, 0, 0), thickness=2)

                # frames = cv2.resize(frame, (1080, 720))
                cv2.imshow("image", clone)

                k = cv2.waitKey(0)

                # 프로그램 종료
                if k == 27:
                    flag = 2
                    break

                # 다음 프레임으로 넘어가기
                if k == ord("d"):
                    makeXML(id, name, blobs)
                    id += 1
                    clicked_points = []
                    blobs = []
                    # cv2.imwrite('C:\MyWorkspace\Make_AIDataset\image_F18005_1_202009140700/deeplearning/' + name, clone)

                    break

                if k == ord('s'):
                    compare(clicked_points, blobs)
                    clicked_points = []

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



