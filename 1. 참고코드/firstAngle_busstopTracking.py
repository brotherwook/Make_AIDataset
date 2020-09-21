# 버스 전용차선 !!!!!!!!!!!!!1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 11:51:00 2017

@author: alexdrake
"""

import cv2
import numpy as np
import time
import logging
import math
import re
from os import walk
import os

# Vehicle_counter from Dan Maesks response on
# https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515


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
        self.log = logging.getLogger("vehicle_counter")

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

    def update_vehicle(self, vehicle, matches):
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
                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        print("No match for vehicle #%d. frames_since_seen=%d" %(
                       vehicle.id, vehicle.frames_since_seen))

        return None

    def update_count(self, matches, output_image=None):
        print("Updating count using %d matches..." % ( len(matches)))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
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

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted and (((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1)) or
                                        ((vehicle.last_position[1] < self.divider) and (
                                            vehicle.vehicle_dir == -1))) and (vehicle.frames_seen > 6):

                vehicle.counted = True
                # update appropriate counter
                if ((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1) and (
                        vehicle.last_position[0] >= (int(frame_w / 2) - 10))):
                    self.vehicle_RHS += 1
                    self.vehicle_count += 1
                elif ((vehicle.last_position[1] < self.divider) and (vehicle.vehicle_dir == -1) and (
                        vehicle.last_position[0] <= (int(frame_w / 2) + 10))):
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

# get working directory
loc = os.path.abspath('')

# Video source
inputFile = 'C:/MyWorkspace/Detectioncode/inputs/000_0820.mp4'

# for testing
tracked_blobs = []
tracked_conts = []
t_retval = []


# inputFile 이름에 정규표현식에 맞는 형식 있으면 camera에 넣기
camera = re.match(r".*/(\d+)_.*", inputFile)
camera = camera.group(1)  # camera는 정규표현식 만족하는 것 중 첫번째
print("camera", camera)
# import video file
cap = cv2.VideoCapture(inputFile)

fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기
delay = int(1000/fps)
# 추적 경로를 그리기 위한 랜덤 색상
color = np.random.randint(0, 255, (200, 3))
lines = None  # 추적 선을 그릴 이미지 저장 변수
prevImg = None
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

# get list of background files
f = []

# backgrounds폴더에 들어있는 모든 파일 리스트 형태로 가져와서 filenames에 넣기
for (_, _, filenames) in walk(loc + "/backgrounds/"):
    f.extend(filenames)  # filenames의 원소들 f의 원소로 넣기
    break

# if background exists for camera: import, else avg will be built on fly

bg = 'C:/MyWorkspace/Detectioncode/backgrounds/222_bg_f_2.jpg'
default_bg = cv2.imread(bg)
type(default_bg)
default_bg = cv2.cvtColor(default_bg, cv2.COLOR_BGR2HSV)  # H 색상
# HSV중 S(채도) = avgSat,V(명도) = default_bg 채널만 사용
(_, avgSat, default_bg) = cv2.split(default_bg)
avg = default_bg.copy().astype("float")

cv2.imshow("default_bg", default_bg)
cv2.imshow("avg",avg)
# get frame size
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# create a mask (manual for each camera)
    
mask = np.zeros((frame_h, frame_w), np.uint8)
mask[:, :] = 255

for i in range(854):
    mask[: int(80 + (365/853) * i), i:] = 0
    mask[int(155 + (375 / 853) * i) - 3:int(155 + (375 / 853) * i) + 3, i] = 0
    mask[int(198 + 20+ (370 / 853) * i) - 1:int(198+ 20 + (370 / 853) * i) + 1, i] = 0
    mask[int(240 + 20+ (370 / 853) * i) - 1:int(240+ 20 + (370 / 853) * i) + 1, i] = 0
    mask[int(278 + 20+ (370 / 853) * i) - 1:int(278+ 20 + (370 / 853) * i) + 1, i] = 0
    mask[int(322 + 20+ (370 / 853) * i) - 1:int(322+ 20 + (370 / 853) * i) + 1, i] = 0
    mask[int(359 + 20+ (370 / 853) * i) - 1:int(359+ 20 + (370 / 853) * i) + 1, i] = 0


cv2.imshow("mask", mask)
# The cutoff for threshold. A lower number means smaller changes between
# the average and current scene are more readily detected.
THRESHOLD_SENSITIVITY = 30  # 40
# t_retval = [] 현재
t_retval.append(THRESHOLD_SENSITIVITY)
# Blob size limit before we consider it for tracking.
CONTOUR_WIDTH = 30  # 사람 탐지하기 위한 폭
CONTOUR_HEIGHT = 30  # 사람을 탐지하기 위한 높이

LIMIT_WIDTH = 200
LIMIT_HEIGHT = 200

BUS_WIDTH = 160
BUS_HEIGHT = 100
# The weighting to apply to "this" frame when averaging. A higher number
# here means that the average scene will pick up changes more readily,
# thus making the difference between average and current scenes smaller.
DEFAULT_AVERAGE_WEIGHT = 0.001  # 0.01 #작을 수록 차량이 잘 감지됨
INITIAL_AVERAGE_WEIGHT = DEFAULT_AVERAGE_WEIGHT / 50
# Blob smoothing function, to join 'gaps' in cars
SMOOTH = max(1, int(round((CONTOUR_WIDTH ** 0.1) / 2, 0)))  # 수정
SMOOTH = 2
# Constants for drawing on the frame.
LINE_THICKNESS = 1
CAR_LINE_THICKNESS = 1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
out = loc + '/outputs/' + camera + '_output.mp4'
out = cv2.VideoWriter(out, fourcc, 20, (frame_w, frame_h))

outblob = loc + '/outputs/' + camera + '_outblob.mp4'
diffop = loc + '/outputs/' + camera + '_outdiff.mp4'
outblob = cv2.VideoWriter(outblob, fourcc, 20, (frame_w, frame_h))
diffop = cv2.VideoWriter(diffop, fourcc, 20, (frame_w, frame_h))

# A list of "tracked blobs".
blobs = []
car_counter = None  # will be created later
frame_no = 0
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_cars = 0

start_time = time.time()
ret, frame = cap.read()  # 비디오 한 프레임씩 읽어오기, 제대로 읽으면 ret=True, frame = 읽은 프레임

while ret:  # 프레임 잘 읽어올
    ret, frame = cap.read()
    frame_no = frame_no + 1

    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임 경우

    if ret and frame_no < total_frames:

        #print("Processing frame ",frame_no)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # only use the Value channel of the frame
        (_, _, grayFrame) = cv2.split(frame)

        # 부드러운 이미지 (src, d, sigmaColor, sigmaSpace)
        grayFrame = cv2.bilateralFilter(grayFrame, 11, 21, 21)
        if avg is None:  # 백그라운드 파일 없는 경우 avg = None이었음
            # Set up the average if this is the first time through.
            avg = grayFrame.copy().astype("float")
            print('test')
            continue

        # Build the average scene image by accumulating this frame
        # with the existing average.
        if frame_no < 10:
            def_wt = INITIAL_AVERAGE_WEIGHT
        else:
            def_wt = DEFAULT_AVERAGE_WEIGHT

        # cv2.accumulateWeighted(grayFrame, avg, def_wt)  # 가중치 적용 화소 누적합 (input, output, 가중치)

        # export averaged background for use in next video feed run
        # if frame_no > int(total_frames * 0.975):
        if frame_no > int(200):
            grayOp = cv2.cvtColor(cv2.convertScaleAbs(
                avg), cv2.COLOR_GRAY2BGR)  # convertScaleAbs : 절대값
            backOut = loc + "/backgrounds/" + camera + "_bg2.jpg"
            cv2.imwrite(backOut, grayOp)  # grayOp를 backout이름으로 저장

        # Compute the grayscale difference between the current grayscale frame and
        # the average of the scene.
        differenceFrame = cv2.absdiff(grayFrame, cv2.convertScaleAbs(avg))

        # blur the difference image
        differenceFrame = cv2.GaussianBlur(differenceFrame, (5, 5), 0)
        cv2.imshow("difference", differenceFrame)
        diffout = cv2.cvtColor(differenceFrame, cv2.COLOR_GRAY2BGR)
        diffop.write(diffout)

        # get estimated otsu threshold level
        retval, _ = cv2.threshold(differenceFrame, 0, 255,  # ( img, 문턱값, 바꿔줄값,flag)
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # add to list of threshold levels
        t_retval.append(retval)

        # apply threshold based on average threshold value
        # 0819 - threshold 0.5 곱해주니까 검은차 잡음
        if frame_no < 10:
            ret2, thresholdImage = cv2.threshold(differenceFrame,
                                                 int(np.mean(t_retval)*0.5),
                                                 255, cv2.THRESH_BINARY)

        else: # 최근 10개 쓰레시홀드
            ret2, thresholdImage = cv2.threshold(differenceFrame,
                                                 int(np.mean(
                                                     t_retval[-10:-1]) * 0.5),
                                                 255, cv2.THRESH_BINARY)

        # We'll need to fill in the gaps to make a complete vehicle as windows
        # and other features can split them!
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (SMOOTH, SMOOTH))  # 커널 생성 (타원모양,크기)

        # Remove noise
        thresholdImage = cv2.morphologyEx(
            thresholdImage, cv2.MORPH_OPEN, kernel)

        thresholdImage = cv2.erode(
            thresholdImage, kernel, iterations=3)

        # Fill any small holes
        # thresholdImage = cv2.morphologyEx(
        #     thresholdImage, cv2.MORPH_CLOSE, kernel, iterations=1)

        # thresholdImage = cv2.morphologyEx(
        #     thresholdImage, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Dilate to merge adjacent blobs
        # thresholdImage = cv2.dilate(
        #     thresholdImage, kernel, iterations=1)  # 이미지 팽창

        # apply mask # mask의 0이 아닌 부분만 이미지와 연산 ,0(검정)인 부분은 마스크 그대로
        thresholdImage = cv2.bitwise_and(
            thresholdImage, mask)  # 비트연산 수헹

        cv2.imshow('thresholdImage', thresholdImage)

        threshout = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        outblob.write(threshout)

        cv2.imshow('threshout', threshout)

        # Find contours aka blobs in the threshold image.
        #        _, contours, hierarchy = cv2.findContours(thresholdImage,
        #                                                  cv2.RETR_EXTERNAL,
        #                                                  cv2.CHAIN_APPROX_SIMPLE) # 컨투어 라인을 그릴 수 있는 포인트만 저장
        contours, hierarchy = cv2.findContours(thresholdImage,
                                               cv2.RETR_EXTERNAL,  # 가장 바깥쪽 라인만 찾음
                                               cv2.CHAIN_APPROX_SIMPLE)  # 컨투어 라인을 그릴 수 있는 포인트만 저장

        test_img = np.ones_like(frame)

        #        print("Found ",len(contours)," vehicle contours.")
        # process contours if they exist!
        if contours:
            for (i, contour) in enumerate(contours):
                # Find the bounding rectangle and center for each blob
                # 컨투어라인을 둘러싸는 사각형 그리기( 회전 고려하지 않음)
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = ((w > CONTOUR_WIDTH) and (h > CONTOUR_HEIGHT) and (
                    w < LIMIT_WIDTH) and (h < LIMIT_HEIGHT))

                #                print("Contour #",i,": pos=(x=",x,", y=",y,") size=(w=",w,
                #                      ", h=",h,") valid=",contour_valid)

                if not contour_valid:
                    continue

                center = (int(x + w / 2), int(y + h / 2))
                blobs.append(((x, y, w, h), center))

        for (i, match) in enumerate(blobs):
            contour, centroid = match
            x, y, w, h = contour

            # store the contour data
            c = dict(
                frame_no=frame_no,
                centre_x=x,
                centre_y=y,
                width=w,
                height=h
            )
            tracked_conts.append(c)
            if ((((205/482) * (x + w - 1) + 128) > y) and (w < 100) and (h < 100)):
                cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1),
                              (0, 250, 190), 2 * CAR_LINE_THICKNESS)
            else:
                cv2.rectangle(frame, (x, y), (x + w - 1, y +
                                              h - 1), (0, 0, 255), CAR_LINE_THICKNESS)

       # cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), CAR_LINE_THICKNESS)

            # cv2.circle(frame, centroid, 2, (0, 0, 255), -1)  # 자동차 이동 추적하는 원

        if car_counter is None:
            print("Creating vehicle counter...")
            car_counter = VehicleCounter(
                frame.shape[:2], 7 * frame.shape[0] / 10)
        #            car_counter = VehicleCounter(frame.shape[:2], 3*frame.shape[0] / 3)

        # get latest count
        car_counter.update_count(blobs, frame)
        current_count = car_counter.vehicle_RHS + car_counter.vehicle_LHS

        # print elapsed time to console
        elapsed_time = time.time() - start_time
        #        print("-- %s seconds --" % round(elapsed_time,2))

        # output video
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        # draw dividing line 버스 전용차선 그리기
        # cv2.line(frame, (0, 158), (316, 293),
        #          (0, 255, 0), 4*LINE_THICKNESS)
        # cv2.line(frame, (316, 293), (492, 396),
        #          (0, 255, 0), 4 * LINE_THICKNESS)
        #
        # cv2.line(frame, (0, 128), (482, 333),
        #          (0, 0, 255), 4 * LINE_THICKNESS)
        # update with latest count
        total_cars = current_count

        # draw upper limit
#        cv2.line(frame, (0, 100), (frame_w, 100), (0, 0, 0), LINE_THICKNESS)
        #frame = cv2.add(frame, lines)
        cv2.imshow("preview", frame)
        out.write(frame)

        if cv2.waitKey(0) == 27:  # 27ms 동안 키 입력 대기
            break
    else:
        break

# cv2.line()
cv2.destroyAllWindows()
cap.release()
out.release()
