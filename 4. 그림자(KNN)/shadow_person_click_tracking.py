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


def GetArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Enter the image files path")
    ap.add_argument("--sampling", default=1, help="Enter the sampling number.(default = 1)")
    args = vars(ap.parse_args())
    path = args['path']
    sampling = int(args['sampling'])
    return path, sampling

#%%
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

    def add_position(self, new_position):
        self.positions.append(new_position)
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
        self.max_unseen_frames = 10

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

    @staticmethod
    def is_valid_vector(a, b):
        # vector is only valid if threshold distance is less than 12
        # and if vector deviation is less than 30 or greater than 330 degs
        distance, angle, _, _ = a
        threshold_distance = 20.0
        return (distance <= threshold_distance)

    def update_vehicle(self, vehicle, matches, t):
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
                vehicle.add_position(centroid)
                vehicle.frames_seen += 1
                # check vehicle direction
                if vector[3] > 0:
                    # positive value means vehicle is moving DOWN
                    vehicle.vehicle_dir = 1
                elif vector[3] < 0:
                    # negative value means vehicle is moving UP
                    vehicle.vehicle_dir = -1
                print("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)" % (
                               centroid[0], centroid[1], vehicle.id, vector[0], vector[1]))

                # 지워도 되는코딩
                # f = open('C:/MyWorkspace/Detectioncode/temp/' + str(vehicle.id) + ".txt", "a+")
                # f.write('   <box frame="' + str(t) + '" outside="0" occluded="0" keyframe="1"'
                #                                             'xtl="'+ str(contour[0]) +'" ytl="'+ str(contour[1]) +
                #         '" xbr="'+ str(contour[2]) +'" ybr="'+ str(contour[3]) + '"></box>' + '\n')
                # f.close()

                track = root.find('track[@id="' + str(vehicle.id) + '"]')
                outside = 0
                if contour[0] < 10 and contour[1] <10 and (contour[0]+contour[2]) > (width-10) and (contour[1] + contour[3]) > (height-10):
                    print("nononnono")
                    outside = 1
                if track is not None:
                    # searches with xml attributes must have '@' before the name
                    box = SubElement(track, 'box')
                    box.attrib["frame"] = str(t)  # 박스의 프레임 넘버
                    box.attrib["outside"] = str(outside)
                    box.attrib["occluded"] = '0'
                    box.attrib["keyframe"] = '1'
                    box.attrib["xtl"] = str(contour[0])
                    box.attrib["ytl"] = str(contour[1])
                    box.attrib["xbr"] = str(contour[0] + contour[2])
                    box.attrib["ybr"] = str(contour[1] + contour[3])
                else:
                    print(track)

                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        print("No match for vehicle #%d. frames_since_seen=%d" %(
                       vehicle.id, vehicle.frames_since_seen))

        return None

    def update_count(self, matches, t, output_image=None):
        global track
        print("Updating count using %d matches..." % ( len(matches)))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches, t)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            print("Created new vehicle #%d from match (%d, %d)." %(
                           new_vehicle.id, centroid[0], centroid[1]))

            # 지워도 되는코딩
            # f = open('C:/MyWorkspace/Detectioncode/temp/' + str(new_vehicle.id) + ".txt", "w")
            # f.write("<track id=" + str(new_vehicle.id) + ' label="차량" source="manual">' + '\n')
            # f.close()

            track = SubElement(root, 'track')
            track.attrib["id"] = str(new_vehicle.id)  # 차량 id\n"
            track.attrib["label"] = 'person'
            track.attrib["source"] = 'manual'

            box = SubElement(track, 'box')
            box.attrib["frame"] = str(t)  # 박스의 프레임 넘버
            box.attrib["outside"] = '0'
            box.attrib["occluded"] = '0'
            box.attrib["keyframe"] = '1'
            box.attrib["xtl"] = str(contour[0])
            box.attrib["ytl"] = str(contour[1])
            box.attrib["xbr"] = str(contour[0] + contour[2])
            box.attrib["ybr"] = str(contour[1] + contour[3])

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

                print("Counted vehicle #%d (total count=%d)." % (
                               vehicle.id, self.vehicle_count))

        # Remove vehicles that have not been seen long enough
        removed = [v.id for v in self.vehicles
                   if v.frames_since_seen >= self.max_unseen_frames]
        self.vehicles[:] = [v for v in self.vehicles
                            if not v.frames_since_seen >= self.max_unseen_frames]
        for id in removed:
            print("Removed vehicle #%d." % (id))

        print("Count updated, tracking %d vehicles." % (
                       len(self.vehicles)))


# ============================================================================



#%%
# 영상불러오기
cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009140900.avi')
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
#%%
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
                print("(" + str(points[1]) + ", " + str(points[0]) + ')')
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

    else:
        if len(roi[0]) > 0:
            # 마스크가 생성됬을 경우 마스크 처리한 부분만 조사
            cv2.imshow("mask", mask)
            roi_img = cv2.bitwise_and(fgmask, mask)
            ret, bImage = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
        else:
            # 마스크를 따로 안만들면 영상 전체 조사
            ret, bImage = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            
        cv2.imshow("b", bImage)
        contours, hierarchy = cv2.findContours(bImage, mode, method)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if AREA_TH < area < 1000:
                print(area)
                x, y, width, height = cv2.boundingRect(cnt)
                if width < 25:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

                center = (int(x + width / 2), int(y + height / 2))
                blobs.append(((x, y, width, height), center))

        if car_counter is None:
            print("Creating vehicle counter...")
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
tree = ElementTree(root)
tree.write('annotations.xml')  #xml 파일로 보내기
