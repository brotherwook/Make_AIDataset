import cv2
import numpy as np
import math
from xml.etree.ElementTree import Element, SubElement, ElementTree

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
        threshold_distance = 12.0
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

                if track is not None:
                    # searches with xml attributes must have '@' before the name
                    box = SubElement(track, 'box')
                    box.attrib["frame"] = str(t)  # 박스의 프레임 넘버
                    box.attrib["outside"] = '0'
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
            track.attrib["label"] = 'car'
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

        # Optionally draw the vehicles on an image
        # if output_image is not None:
            # for vehicle in self.vehicles:
            #    vehicle.draw(output_image)

            # # LHS
            # cv2.putText(output_image, ("LH Lane: %03d" % self.vehicle_LHS),
            #             (12, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (127, 255, 255), 2)
            # # RHS
            # cv2.putText(output_image, ("RH Lane: %03d" % self.vehicle_RHS),
            #             (216, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (127, 255, 255), 2)

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
background_path = "C:/MyWorkspace/Detectioncode/temp/F18006_2_202009112100_bgr.png"

bkg_bgr = cv2.imread(background_path)
bkg_gray = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('C:\MyWorkspace\Detectioncode\inputs\F18006_2\F18006_2_202009112100.avi')
if (not cap.isOpened()):
    print('Error opening video')


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

while True:
    try:
        retval, frame = cap.read()
        if not retval:
            break
        t += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff_gray = cv2.absdiff(gray, bkg_gray)
        diff_bgr = cv2.absdiff(frame, bkg_bgr)

        db, dg, dr = cv2.split(diff_bgr)
        ret, bb = cv2.threshold(db, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, bg = cv2.threshold(dg, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, br = cv2.threshold(dr, TH, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        bImage = cv2.bitwise_or(bb, bg)
        bImage = cv2.bitwise_or(br, bImage)
        bImage = cv2.medianBlur(bImage, 5)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        bImage = cv2.morphologyEx(bImage,cv2.MORPH_OPEN, k)

        # bImage = cv2.erode(bImage, None, 5)
        # bImage = cv2.dilate(bImage, None, 5)
        # bImage = cv2.erode(bImage, None, 7)
        cv2.imshow("b", bImage)
        contours, hierarchy = cv2.findContours(bImage, mode, method)
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > AREA_TH:
                print(area)
                x, y, width, height = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

                center = (int(x + width / 2), int(y + height / 2))
                blobs.append(((x, y, width, height), center))

        cv2.imshow("frame", frame)
        cv2.imshow("bImage", bImage)
        cv2.imshow("diff_gray", diff_gray)
        cv2.imshow("diff_bgr", diff_bgr)

        if car_counter is None:
            print("Creating vehicle counter...")
            car_counter = VehicleCounter(
                frame.shape[:2], 7 * frame.shape[0] / 10)
        #            car_counter = VehicleCounter(frame.shape[:2], 3*frame.shape[0] / 3)

        # get latest count
        car_counter.update_count(blobs, t, frame)
        current_count = car_counter.vehicle_RHS + car_counter.vehicle_LHS

        if cv2.waitKey(1) == 27:
            break

    except KeyboardInterrupt:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
tree = ElementTree(root)
tree.write('annotations.xml')  #xml 파일로 보내기