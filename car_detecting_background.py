import cv2
import numpy as np


#%% 사람 클래스
class Person:
    def __init__(self, position, startframe, id):
        self.positions = [position]
        self.t = [startframe]
        self.startframe = startframe
        self.updateframe = startframe
        # self.mean = mean # 안쓰면 지울것
        self.id = id
        # track = SubElement(root, 'track')
        # track.attrib["id"] = str(id)  # 차량 id\n"
        # track.attrib["label"] = 'person'
        # track.attrib["source"] = 'manual'

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
                cv2.putText(frame, ("person %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2, (127, 255, 255), 2)
        else:
            for i, center in enumerate(blobs):
                for j, person in enumerate(self.people):
                    k = person.update_position(center, t)
                    if k != -1:
                        cv2.putText(frame, ("person %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (127, 255, 255), 2)
                        break

                if k == -1:
                    person = Person(center, t, self.cntperson)
                    self.people.append(person)
                    self.cntperson += 1
                    cv2.putText(frame, ("person %d" % person.id), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                (127, 255, 255), 2)


            for j, person in enumerate(self.people):
                if t - person.updateframe > 200:
                    removed.append(person.id)

            # self.removePerson(removed)
            removed = []

            self.people[:] = [person for person in self.people
                                if t - person.updateframe <= 50]

    # def removePerson(self, removed):
        # for i, v in enumerate(removed):
        #     makeXML(removed[i], self.people)


#%% 영상 전처리
def imagepreprocessing(frame, img=None):

    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        diff_gray = cv2.absdiff(gray, bkg_gray)
        diff_bgr = cv2.absdiff(img, bkg_bgr)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff_gray = cv2.absdiff(gray, bkg_gray)
        diff_bgr = cv2.absdiff(frame, bkg_bgr)

    ret, gr = cv2.threshold(diff_gray, TH, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    db, dg, dr = cv2.split(diff_bgr)
    ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bImage = cv2.bitwise_or(bb, bg)
    bImage = cv2.bitwise_or(br, bImage)

    # 노이즈 제거
    medianblur = cv2.medianBlur(bImage, 5)
    cv2.imshow("median", medianblur)

    # 팽창연산
    # k = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 10))
    # dil = cv2.dilate(medianblur, k)
    # cv2.imshow("dil", dil)

    # 윤곽선찾기
    contours, hierachy = cv2.findContours(medianblur, mode, method)

    # 윤곽선 안을 흰색으로 채우기
    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(medianblur, contour, 255)

    # 다시한번 팽창 연산
    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 10))
    # dil = cv2.dilate(dil, k)

    # 다시 윤곽선 찾기
    contours, hierachy = cv2.findContours(dil, mode, method)

    # 중심점 구하기
    for i, contour in enumerate(contours):
        cv2.fillPoly(dil, contour, 255)
        area = cv2.contourArea(contour)
        if AREA_TH < area:
            x, y, width, height = cv2.boundingRect(contour)
            center = (int(x + width / 2), int(y + height / 2))
            cv2.drawContours(frame, contour, -1, (0, 0, 255), 1)
            cv2.rectangle(frame,(x,y),(x+width, y+height), (255,0,0))
            cv2.putText(frame, "area : " + str(area) + ", width,height : " + str(width) + "," + str(height),(x+int(width/2), y+int(height/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)
            blobs.append(center)

    cv2.imshow("aa", dil)

    return dil


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
            cv2.circle(image, (point[0], point[1]), 2, (0, 255, 255), thickness=-1)
        cv2.imshow("image", image)

# 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", MouseLeftClick)

#%%
background_path = "C:/MyWorkspace/Make_AIDataset/backgrounds/F20001;3_3sxxxx0_bgr.png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

# 배경이미지와 동영상의 크기 맞추기
# bkg_bgr = cv2.resize(bkg_bgr, (int(bkg_bgr.shape[1]/2), int(bkg_bgr.shape[0]/2)))
# bkg_gray = cv2.resize(bkg_gray, (int(bkg_gray.shape[1]/2), int(bkg_gray.shape[0]/2)))

cap = cv2.VideoCapture('C:\MyWorkspace\Make_AIDataset\inputs\F20001;3_3sxxxx0;양재1동 23;양재환승센터;(Ch 01)_[20200928]080000-[20200928]080600(20200928_080000).avi')
if (not cap.isOpened()):
    print('Error opening video')


height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

t = 0
TH = 0  # Binary Threshold
AREA_TH = 900  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

flag = 0
mask = None
roi = [[]]
temp = []

blobs = []

margin = 20
person_counter = None
removed = []

#%%
while True:
    try:
        retval, frame = cap.read()
        if not retval:
            break
        t += 1
        frame = cv2.resize(frame, (bkg_bgr.shape[1], bkg_bgr.shape[0]))

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

                if key == ord('l'):
                    if len(clicked_points) > 0:
                        for i in range(len(clicked_points)-1):
                            cv2.line(mask,clicked_points[i],clicked_points[i+1], 0, 2, cv2.LINE_AA)
                        clicked_points = []

                if key == ord('r'):
                    cv2.destroyAllWindows()
                    clicked_points = []
                    flag = 1
                    break

        if flag == 1:
            if len(roi[0]) > 0:
                # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
                cv2.imshow("mask", mask)
                roi_img = cv2.bitwise_and(frame, mask)
                bkg_bgr = cv2.bitwise_and(bkg_bgr, mask)
                dil = imagepreprocessing(frame, roi_img)

                for j in range(len(temp)):
                    for i, v in enumerate(roi[0]):
                        if i < len(roi[0]) - 1:
                            frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][i + 1]), (0, 0, 255), 2)
                        else:
                            frame = cv2.line(frame, tuple(temp[j][i]), tuple(temp[j][0]), (0, 0, 255), 2)
            else:
                # 마스크를 따로 안만들면 영상 전체 조사
                dil = imagepreprocessing(frame)

            if person_counter is None:
                person_counter = PersonCounter()

            person_counter.update_people(blobs, t, frame)

            blobs = []

            cv2.imshow("frame", frame)

        if cv2.waitKey(1) == 27:
            break

    except KeyboardInterrupt:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()