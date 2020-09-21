import numpy as np
import cv2
import argparse
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

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
root = Element('annotations')
track = None

#%%
# ============================================================================

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.framet = [t]
        self.frames_since_seen = 0
        self.frames_seen = 0
        self.counted = False
        self.vehicle_dir = 0

    @property
    def last_position(self):
        return self.positions[-1]

    @property
    def last_position2(self):
        return self.positions[-2]

    def add_position(self, new_position, t):
        self.positions.append(new_position)
        self.framet.append(t)
        self.frames_since_seen = 0
        self.frames_seen += 1

    def draw(self, output_image):
        for point in self.positions:
            cv2.circle(output_image, point, 2, (0, 0, 255), -1)
            cv2.polylines(output_image, [np.int32(self.positions)]  # 다각형그리기
                          , False, (0, 0, 255), 1)

# ============================================================================


class VehicleCounter(object):
    def __init__(self, shape, divider):
        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.vehicle_LHS = 0
        self.vehicle_RHS = 0
        self.max_unseen_frames = 30

    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values decrease in clockwise direction.
        """

        iou = caliou([(a[0] - (h_width/2)), (a[1] - (h_height/2)), (a[0] + (h_width/2)), (a[1] + (h_height/2))],
                     [(b[0] - (h_width/2)), (b[1] - (h_height/2)), (b[0] + (h_width/2)), (b[1] + (h_height/2))])

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

        return distance, angle, dx, dy, iou

    @staticmethod
    def is_valid_vector(a, b):
        # vector is only valid if threshold distance is less than 12
        # and if vector deviation is less than 30 or greater than 330 degs
        distance, angle, _, _, iou = a
        threshold_distance = 20.0
        threshold_iou = 0.8

        if (distance <= threshold_distance) and (iou >= threshold_iou):
            print(iou)
            return True
        else:
            False

    def update_vehicle(self, vehicle, matches, t, frame):
        global track
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid = match

            # store the vehicle data
            vector = self.get_vector(vehicle.last_position, centroid)

            # only measure angle deviation if we have enough points
            if vehicle.frames_seen > 2:
                prevVector = self.get_vector(
                    vehicle.last_position2, vehicle.last_position)
                angleDev = abs(prevVector[1] - vector[1])
                # if vehicle.id == 0:
                #     print("preV : " + str(prevVector[0]) + ", nowV : " + str(vector[0]))
            else:
                angleDev = 0

            b = dict(
                id=vehicle.id,
                center_x=centroid[0],
                center_y=centroid[1],
                vector_x=vector[0],
                vector_y=vector[1],
                dx=vector[2],
                dy=vector[3],
                counted=vehicle.counted,
                frame_number=frame_no,
                angle_dev=angleDev
            )

            tracked_blobs.append(b)

            # check validity
            if self.is_valid_vector(vector, angleDev):
                xtl, ytl, xbr, ybr = int(centroid[0] - h_width / 2), int(centroid[1] - h_height / 2), int(centroid[0] + h_width / 2), int(centroid[1] + h_height / 2)
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)
                vehicle.add_position(centroid, t)
                vehicle.frames_seen += 1
                # check vehicle direction
                if vector[3] > 0:
                    # positive value means vehicle is moving DOWN
                    vehicle.vehicle_dir = 1
                elif vector[3] < 0:
                    # negative value means vehicle is moving UP
                    vehicle.vehicle_dir = -1
                # print("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)" % (
                #                centroid[0], centroid[1], vehicle.id, vector[0], vector[1]))

                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        # print("No match for vehicle #%d. frames_since_seen=%d" %(
        #                vehicle.id, vehicle.frames_since_seen))

        return None

    def update_count(self, matches, t, output_image=None):
        global track
        # print("Updating count using %d matches..." % ( len(matches)))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches, t, output_image)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            # print("Created new vehicle #%d from match (%d, %d)." %(
            #                new_vehicle.id, centroid[0], centroid[1]))


            track = SubElement(root, 'track')
            track.attrib["id"] = str(new_vehicle.id)  # 차량 id\n"
            track.attrib["label"] = 'person'
            track.attrib["source"] = 'manual'

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted and (((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1)) or
                                        ((vehicle.last_position[1] < self.divider) and (
                                            vehicle.vehicle_dir == -1))) and (vehicle.frames_seen > 6):

                vehicle.counted = True
                # update appropriate counter
                if ((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1) and (
                        vehicle.last_position[0] >= (int(width / 2) - 10))):
                    self.vehicle_RHS += 1
                    self.vehicle_count += 1
                elif ((vehicle.last_position[1] < self.divider) and (vehicle.vehicle_dir == -1) and (
                        vehicle.last_position[0] <= (int(width / 2) + 10))):
                    self.vehicle_LHS += 1
                    self.vehicle_count += 1

                # print("Counted vehicle #%d (total count=%d)." % (
                #                vehicle.id, self.vehicle_count))

        # Remove vehicles that have not been seen long enough
        removed = [v.id for v in self.vehicles
                   if v.frames_since_seen >= self.max_unseen_frames]
        #
        for i, v in enumerate(self.vehicles):
            if v.frames_since_seen >= self.max_unseen_frames:
                makeXML(v.id, v.positions, v.framet)

        self.vehicles[:] = [v for v in self.vehicles
                            if not v.frames_since_seen >= self.max_unseen_frames]
        # for id in removed:
        #     print("Removed vehicle #%d." % (id))
        #
        # print("Count updated, tracking %d vehicles." % (
        #                len(self.vehicles)))


# ============================================================================

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

def makeXML(id, centroid, framet):
    global root
    global track

    track = root.find('track[@id="' + str(id) + '"]')
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


#%% iou 계산
def caliou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    area_intersection = (x2 - x1) * (y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_union = area_box1 + area_box2 - area_intersection

    if area_union == 0 or area_intersection == 0:
        return 0
    iou = area_intersection / area_union
    if iou <0:
        return 0

    return iou

#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009140800.avi')
if (not cap.isOpened()):
    print('Error opening video')

#
fgbg = cv2.createBackgroundSubtractorKNN()

#%%
# A list of "tracked blobs".
blobs = []
car_counter = None  # will be created later
frame_no = 0
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_cars = 0

# for testing
tracked_blobs = []
tracked_conts = []
t_retval = []

#%%
height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
t = 0
TH = 0  # Binary Threshold
AREA_TH = 80  # Area Threshold


mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

flag = 0
mask = None
roi = [[]]

h_width = 20
h_height = 50



#%%

initXML()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    # 처음 4프레임 데이터쌓기
    if t < 4:
        continue


    cv2.imshow("fgmask", fgmask)
    # 마스크 설정 부분
    # 마우스로 보기를 원하는 부분 클릭하고 n누르면 해당부분만 확인
    # 원하는 부분 클릭후  다음 프레임에서 또 다시 클릭하면 모두 확인가능
    # r 누르면 영상 재생
    if flag == 0:
        clone = frame.copy()
        cv2.imshow("image", frame)
        key = cv2.waitKey(0)

        if mask is None:
            mask = np.zeros_like(fgmask)
        
        #클릭한거 저장 및 마스크 생성
        if key == ord('n'):
            roi = [[]]
            for points in clicked_points:
                # print("(" + str(points[1]) + ", " + str(points[0]) + ')')
                roi[0].append((points[1], points[0]))
            if len(roi[0]) > 0:
                roi = np.array(roi)
                cv2.fillPoly(mask, roi, (255,255,255))
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

    # object detect한 영상 재생
    else:
        if len(roi[0]) > 0:
            # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
            cv2.imshow("mask", mask)
            roi_img = cv2.bitwise_and(fgmask, mask)
            ret, bImage = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            ret, bImage = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        # 커널 생성
        e = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

        # bImage = cv2.erode(bImage, e)
        # bImage = cv2.dilate(bImage, k)

        contours, hierarchy = cv2.findContours(bImage, mode, method)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if AREA_TH < area < 1000:
                # print(area)
                x, y, width, height = cv2.boundingRect(cnt)
                center = (int(x + width / 2), int(y + height / 2))
                xtl, ytl, xbr, ybr = x, y, int(x + h_width), int(y + h_height)
                if width < 30:
                    blobs.append(((xtl, ytl, h_width, h_height), center))

        cv2.imshow("b", bImage)

        # 처음에 받은 클래스 사용
        if car_counter is None:
            # print("Creating vehicle counter...")
            car_counter = VehicleCounter(
                frame.shape[:2], 7 * frame.shape[0] / 10)
        #            car_counter = VehicleCounter(frame.shape[:2], 3*frame.shape[0] / 3)

        # get latest count
        car_counter.update_count(blobs, t, frame)
        current_count = car_counter.vehicle_RHS + car_counter.vehicle_LHS

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()

for i, vehicle in enumerate(car_counter.vehicles):
    makeXML(vehicle.id, vehicle.positions, vehicle.framet)

tree = ElementTree(root)
tree.write('annotations.xml')  #xml 파일로 보내기
