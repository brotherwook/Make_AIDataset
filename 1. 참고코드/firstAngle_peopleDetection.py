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
# re: 정규표현식
import re
from os import walk
import os

# Vehicle_counter from Dan Maesks response on
# https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515

# ============================================================================
# contour = (x,y,w,h), centroid = (int(x + w/2), int(y + h/2))
# contour, centroid = match
# new_vehicle = Vehicle(self.next_vehicle_id, centroid) 로 호출


class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.frames_seen = 0
        self.counted = False
        self.vehicle_dir = 0

    # vehicle의 중심점 반환 ex) (4,3)
    @property
    def last_position(self):
        return self.positions[-1]
    # vehicle의 x,y좌표, 가로세로 길이 반환 ex) (1,5,3,2)

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
            cv2.polylines(output_image, [np.int32(
                self.positions)], False, (0, 0, 255), 1)

# ============================================================================

# VehicleCounter(frame.shape[:2], 7*frame.shape[0] / 10) 로 호출
# frame.shape[:2]는 frame의 (세로, 가로) tuple 반환


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

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
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
                angleDev = abs(prevVector[1]-vector[1])
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
                self.log.debug("Added match (%d, %d) to vehicle #%d. vector=(%0.2f,%0.2f)",
                               centroid[0], centroid[1], vehicle.id, vector[0], vector[1])
                return i

        # No matches fit...
        vehicle.frames_since_seen += 1
        self.log.debug("No match for vehicle #%d. frames_since_seen=%d",
                       vehicle.id, vehicle.frames_since_seen)

        return None

# car_counter.update_count(blobs, frame) 로 호출
# center = (int(x + w/2), int(y + h/2))
# blobs에는 ((x, y, w, h), center) 형태로 object 정보가 있다.
    def update_count(self, matches, output_image=None):
        self.log.debug("Updating count using %d matches...", len(matches))

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            # contour = (x,y,w,h), centroid = (int(x + w/2), int(y + h/2))
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            # print("Created new vehicle {} from match ({}, {}).".format(
            #     new_vehicle.id, centroid[0], centroid[1]))
            self.log.debug("Created new vehicle #%d from match (%d, %d).",
                           new_vehicle.id, centroid[0], centroid[1])

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            # count한 vehicle인지 판별 and
            # if not vehicle.counted and (((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1)) or
            #                             ((vehicle.last_position[1] < self.divider) and (vehicle.vehicle_dir == -1))) and (vehicle.frames_seen > 6):
            
            # if not vehicle.counted and (vehicle.last_position[1] > (18/43) * vehicle.last_position[0]) and (vehicle.last_position[1] < 30 + (18/43) * vehicle.last_position[0]):
            if not vehicle.counted and ((18/43)*vehicle.last_position[0] > 160) and ((18/43)*vehicle.last_position[0] < 340):
                self.vehicle_count += 1
                vehicle.counted = True
                # update appropriate counter
                # if ((vehicle.last_position[1] > self.divider) and (vehicle.vehicle_dir == 1) and (vehicle.last_position[0] >= (int(frame_w/2)-10))):
                #     self.vehicle_RHS += 1
                #     self.vehicle_count += 1
                # elif ((vehicle.last_position[1] < self.divider) and (vehicle.vehicle_dir == -1) and (vehicle.last_position[0] <= (int(frame_w/2)+10))):
                #     self.vehicle_LHS += 1
                #     self.vehicle_count += 1
                self.log.debug("Counted vehicle #%d (total count=%d).",
                               vehicle.id, self.vehicle_count)

        # Optionally draw the vehicles on an image
        # if output_image is not None:
        #     추적용 흰점 그리기
        #     for vehicle in self.vehicles:
        #         vehicle.draw(output_image)

        #     LHS
        #     cv2.putText(output_image, ("LH Lane: %03d" % self.vehicle_LHS),
        #                 (12, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (127, 255, 255), 2)
        #     RHS
        #     cv2.putText(output_image, ("RH Lane: %03d" % self.vehicle_RHS),
        #                 (216, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (127, 255, 255), 2)

            # Remove vehicles that have not been seen long enough
        removed = [v.id for v in self.vehicles
                   if v.frames_since_seen >= self.max_unseen_frames]
        self.vehicles[:] = [v for v in self.vehicles
                            if not v.frames_since_seen >= self.max_unseen_frames]
        for id in removed:
            self.log.debug("Removed vehicle #%d.", id)

        self.log.debug("Count updated, tracking %d vehicles.",
                       len(self.vehicles))

# ============================================================================

# get working directory
loc = os.path.abspath('')

# Video source
inputFile = 'C:/MyWorkspace/Detectioncode/inputs/334_0420.mp4'

# for testing
tracked_blobs = []
tracked_conts = []
t_retval = []

# re.match(pattern, string, flags)
# string의 처음부터 시작하여 pattern이 일치되는 것이 있는지 확인 / return: re.Match object
# re.Match.group(n) : 입력받은 객체 내의 값에서 ()기준으로 n번째 그룹 반환
# 여기에선 파일명(456_202004301300)이 camera에 들어가겠군
camera = re.match(r".*/(\d+)_.*", inputFile)
camera = camera.group(1)

# import video file
cap = cv2.VideoCapture(inputFile)

# get list of background files
# os.walk : 모든 파일/폴더 출력하기
f = []
for (_, _, filenames) in walk(loc+"/backgrounds/"):
    # append: 리스트 자체를 원소로 넣는다
    # extend: 리스트면 iterable로 각 항목들을 넣는다
    f.extend(filenames)
    break

# if background exists for camera: import, else avg will be built on fly

# 배경이미지 불러오기
bg = 'C:/MyWorkspace/Detectioncode/backgrounds/334_bg.jpg'
# 배경이미지 읽기
default_bg = cv2.imread(bg)
# 배경이미지 색상 변경 cv2.cvtColor(변경할 이미지, 변경할 색상 옵션)
# BGR: blue, green, red / HSV: Hue(색상), Saturation(채도), Value(진하기)
# return: 3차원 배열. 숫자로 채워진   / HSV 순으로 채워짐
default_bg = cv2.cvtColor(default_bg, cv2.COLOR_BGR2HSV)
(_, avgSat, default_bg) = cv2.split(default_bg)
avg = default_bg.copy().astype("float")

# get frame size
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create a mask (manual for each camera)
mask = np.zeros((frame_h, frame_w), np.uint8)
mask[:, :] = 255

# 차 도로 마스킹
for i in range(600):
    mask[int(120 + (250/600) * i):, : i] = 0

# 인도 상단 도로 마스킹
for i in range(854):
    mask[: int(35 + (355/854) * i), i:] = 0

# 오른쪽 지하철입구, 도로 마스킹하기
for i in range(50):
    mask[400:, 400 + i:] = 0

#cv2.imshow("mask",mask)
# The cutoff for threshold. A lower number means smaller changes between
# the average and current scene are more readily detected.
THRESHOLD_SENSITIVITY = 40  # 40
t_retval.append(THRESHOLD_SENSITIVITY)
# Blob size limit before we consider it for tracking.
CONTOUR_WIDTH = 10  # 자동차를 탐지하기 위한 폭
CONTOUR_HEIGHT = 10  # 자동차를 탐지하기 위한 높이 # 40이면 456파일에서 경차 탐색 안

CONTOUR_WIDTH_LIMIT = 50
CONTOUR_HEIGHT_LIMIT = 40
# The weighting to apply to "this" frame when averaging. A higher number
# here means that the average scene will pick up changes more readily,
# thus making the difference between average and current scenes smaller.
DEFAULT_AVERAGE_WEIGHT = 0.001  # 0.01 #작을 수록 차량이 잘 감지됨
INITIAL_AVERAGE_WEIGHT = DEFAULT_AVERAGE_WEIGHT / 50
# Blob smoothing function, to join 'gaps' in cars
# 궁금증: SMOOTH의 역할?
SMOOTH = max(2, int(round((CONTOUR_WIDTH**0.5)/2, 0)))
# Constants for drawing on the frame.
LINE_THICKNESS = 1
CAR_LINE_THICKNESS = 1

# VideoWriter_fourcc: Codec정보 mp4v
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 파일명이름
out = loc+'/outputs/'+camera+'_output.mp4'
# VideoWriter(outputFile, fourcc, frame, size): 영상저장
# frame: 초당 저장될 frame, size: 저장될 사이즈
out = cv2.VideoWriter(out, fourcc, 20, (frame_w, frame_h))

# 일단 영상 만들어놓기 - 나중에 뒤에서 write
outblob = loc+'/outputs/'+camera+'_outblob.mp4'
diffop = loc+'/outputs/'+camera+'_outdiff.mp4'
outblob = cv2.VideoWriter(outblob, fourcc, 20, (frame_w, frame_h))
diffop = cv2.VideoWriter(diffop, fourcc, 20, (frame_w, frame_h))

# A list of "tracked blobs".
blobs = []
car_counter = None  # will be created later
frame_no = 0
o_q_cnt = 0
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
total_cars = 0

start_time = time.time()
# cap.read(): frame이 없을 경우 false반환
ret, frame = cap.read()

# LK용 변수
# lines = None  #추적 선을 그릴 이미지 저장 변수
prevImg = None
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
color = np.random.randint(0, 255, (200, 3))

while ret:
    ret, frame = cap.read()
    frame_no = frame_no + 1

    # 전체 frame수 만큼 while 반복하기 위한 if문
    if ret and frame_no < total_frames:

        #        print("Processing frame ",frame_no)
        # LK시작
        img_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        # 최초 프레임 경우
        if prevImg is None:
            prevImg = gray
            # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
            lines = np.zeros_like(frame)
            # 추적 시작을 위한 코너 검출  ---①
            prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
        else:
            nextImg = gray
            # 옵티컬 플로우로 다음 프레임의 코너점  찾기 ---②
            nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg,
                                                           prevPt, None, criteria=termcriteria)
            # 대응점이 있는 코너, 움직인 코너 선별 ---③
            prevMv = prevPt[status == 1]
            nextMv = nextPt[status == 1]
            for i, (p, n) in enumerate(zip(prevMv, nextMv)):
                px, py = p.ravel()
                nx, ny = n.ravel()
                # 이전 코너와 새로운 코너에 선그리기 ---④
                # cv2.line(lines, (px, py), (nx,ny), color[i].tolist(), 2)
                # 새로운 코너에 점 그리기
                # cv2.circle(img_draw, (nx, ny), 2, color[i].tolist(), -1)
            # 누적된 추적 선을 출력 이미지에 합성 ---⑤
            frame = cv2.add(img_draw, lines)
            # 다음 프레임을 위한 프레임과 코너점 이월

            prevImg = nextImg
            prevPt = nextMv.reshape(-1, 1, 2)
# LK 끝
        # get returned time
        frame_time = time.time()

        # convert BGR to HSV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # only use the Value channel of the frame
        (_, _, grayFrame) = cv2.split(frame)
        # bilateralFilter: 엣지와 노이즈를 줄여주어 부드러운 영상을 만듦
        # grayFrame = cv2.bilateralFilter(grayFrame, 11, 21, 21)

        # avg: HSV의 V값이 float형태로 넘어옴 (None이면 배경이미지 없음을 의미)
        if avg is None:
            # Set up the average if this is the first time through.
            # 현재 보고있는 frame의 HSV의 V값을 복사하며 avg에 저장
            avg = grayFrame.copy().astype("float")
            continue

        # Build the average scene image by accumulating this frame
        # with the existing average.
        # 1~9번째 frame은 weight로 INITIAL_AVERAGE_WEIGHT 적용
        if frame_no < 10:
            def_wt = INITIAL_AVERAGE_WEIGHT
        else:   # 이후 frame은 weight로 DEFAULT_AVERAGE_WEIGHT 적용
            def_wt = DEFAULT_AVERAGE_WEIGHT
        # grayFrame(src)에 def_wt(weight)을 적용한 값을 avg(dst)에 누적 및 저장
        cv2.accumulateWeighted(grayFrame, avg, def_wt)

        # export averaged background for use in next video feed run
        # 궁금증: if문은 왜 하는거지
        # if frame_no > int(total_frames * 0.975):
        if frame_no > int(200):
            # convertScaleAbs(img): img의 값을 미분하고 절대값을 취하여 값 범위를 8bit unsigned int형으로 바꾼다.
            grayOp = cv2.cvtColor(cv2.convertScaleAbs(avg), cv2.COLOR_GRAY2BGR)
            # backOut = loc+"/backgrounds/"+camera+"_bg.jpg"
            # imwrite(저장될 파일명, 저장할 이미지): 이미지 저장
            # cv2.imwrite(backOut, grayOp)

        # Compute the grayscale difference between the current grayscale frame and
        # the average of the scene.
        # absdiff(원본배열1, 원본배열2) : 배열과 배열의 절대값 차이를 계산
            # gray일 뿐인 grayFrame을 2진화, 투명화 해주는거 같음
        differenceFrame = cv2.absdiff(grayFrame, cv2.convertScaleAbs(avg))
        # blur the difference image
        # 이미지 객체의 윤곽선 블러효과 / 노이즈 줄이기
        differenceFrame = cv2.GaussianBlur(differenceFrame, (5, 5), 0)
#        cv2.imshow("difference", differenceFrame)
        diffout = cv2.cvtColor(differenceFrame, cv2.COLOR_GRAY2BGR)
        diffop.write(diffout)

        # get estimated otsu threshold level
        # cv2.threshold(img, threshold_value, value, flag):
        # img: Grayscale 이미지, threshold_value: 픽셀 문턱값, value: 문턱값보다 크거나 작을때 적용할 값, flag: 문턱값 적용 방법 또는 스타일
        # THRESH_OTSU: 임계값 자동으로 계산
        # retval: 사용된 threshold, _: thresholed image
        retval, _ = cv2.threshold(differenceFrame, 0, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # add to list of threshold levels
        t_retval.append(retval)

        # apply threshold based on average threshold value
        # 궁금증: 왜 이렇게 하는건지는 모르겠다.
        if frame_no < 10:
            ret2, thresholdImage = cv2.threshold(differenceFrame,
                                                 int(np.mean(t_retval)*0.9),
                                                 255, cv2.THRESH_BINARY)
        else:
            ret2, thresholdImage = cv2.threshold(differenceFrame,
                                                 int(np.mean(
                                                     t_retval[-10:-1])*0.9),
                                                 255, cv2.THRESH_BINARY)

        # **Morphological Transformations**
        # 용도: 이미지를 분할하여 단순화, 제거, 보정을 통해 형태를 파악
        # 방법: erosion, dilation, Opening & Closing
            # erosion: 작은 object제거
            # dilation: 경게 부드럽게, 구멍 기
        # We'll need to fill in the gaps to make a complete vehicle as windows
        # and other features can split them!
        # getStructuringElement(element모양, (n, n)): n*n 사이즈의 element 생성해줌 / array 형태로 반환
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SMOOTH, SMOOTH))
        # Fill any small holes
        # dilation적용 후 erosion 적용. 전체적인 윤곽 파악에 적합
        thresholdImage = cv2.morphologyEx(
            thresholdImage, cv2.MORPH_CLOSE, kernel)

        # Remove noise
        # erosion 적용 후 dilation 적용. 작은 object나 돌기 제거에 적합
        thresholdImage = cv2.morphologyEx(
            thresholdImage, cv2.MORPH_OPEN, kernel)

        # 구멍메꾸기, iterations: dilation 적용 반복 횟수
        # Dilate to merge adjacent blobs
        thresholdImage = cv2.dilate(thresholdImage, kernel, iterations=3)

        # apply mask
        # bitwise_and: 비트연산함수, mask범위 내에서 두개의 array를 and(&) 비트연산
        # mask에서 0이 아닌 부분의 색만 유지한 이미지 반환 / 값이 0이면 검정색...
        thresholdImage = cv2.bitwise_and(
            thresholdImage, thresholdImage, mask=mask)
#        cv2.imshow("threshold", thresholdImage)
        threshout = cv2.cvtColor(thresholdImage, cv2.COLOR_GRAY2BGR)
        outblob.write(threshout)

        # Find contours aka blobs in the threshold image.
        # contour란, 동일한 색 또는 동일한 색상 강도를 가진 부분의 가장 자리 경계를 연결한 선
        # 따라서, 이미지에 있는 물체의 모양 분석이나 객체 인식 등에 유용하게 활용되는 도구이다.
        # contour 찾기는 검정 배경에서 흰색 물체를 찾는 것!
#        _, contours, hierarchy = cv2.findContours(thresholdImage,
#                                                  cv2.RETR_EXTERNAL,
#                                                  cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresholdImage,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours[0]))

        # if frame_no % 20 == 0:
        #     contoursCount = len(contours)
        #     print("The number of objects in frame",
        #           str(frame_no), ":", str(contoursCount))

#        print("Found ",len(contours)," vehicle contours.")
        # process contours if they exist!
        # 흰색물체인 자동차의 x,y좌표와 가로세로 길이, 중심좌표를 blobs에 저장
        if contours:
            a = []
            for (i, contour) in enumerate(contours):
                # Find the bounding rectangle and center for each blob
                (x, y, w, h) = cv2.boundingRect(contour)
                contour_valid = (w > CONTOUR_WIDTH) and (h > CONTOUR_HEIGHT) 
                # contour_valid = (w > CONTOUR_WIDTH) and (h > CONTOUR_HEIGHT) and (
                #     w < CONTOUR_WIDTH_LIMIT) and (h < CONTOUR_HEIGHT_LIMIT)
#                print("Contour #",i,": pos=(x=",x,", y=",y,") size=(w=",w,
#                      ", h=",h,") valid=",contour_valid)
                if not contour_valid:
                    continue

                center = (int(x + w/2), int(y + h/2))
                blobs.append(((x, y, w, h), center))
                #길이
                #cv2.putText(frame, str(w), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                #숫자
                # cv2.putText(frame, str(i+1).reverse(), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
#---------------width 평균구하기--------------
                # a.append(w)
                # #print(a)
                # b = np.mean(a)
                # print("mean of w:", )
#---------------------------------------------
        # cv2.line(frame, (180, 160), (610, 340), (0, 255, 0), LINE_THICKNESS)
        # cv2.line(frame, (180, 130), (610, 310), (0, 255, 0), LINE_THICKNESS)

        on_queue = []
        off_queue = []
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

            # cv2.circle(frame, centroid, 2, (0, 0, 255), -1)  # 자동차 이동 추적하는 원
            if centroid[0] > 180 and centroid[1] < (180/430) * centroid[0] + 85 and centroid[1] > (180/430) * centroid[0] + 55:
                cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1),
                              (70, 200, 255), CAR_LINE_THICKNESS)
                # 자동차 이동 추적하는 원
                cv2.circle(frame, centroid, 2, (70, 200, 255), -1)
                on_queue.append(match)
                #사람수 라벨링

                people_number = str(-((i+1)-(len(blobs))))
                # cv2.putText(frame, people_number, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                # cv2.putText(frame, str(w), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                onep = 10
                if (w>onep):
                    for i in range(int(w/onep)-1):
                        cv2.rectangle(frame, (x + onep*(i), y), (x + onep*(i+1) - 1, y + h - 1),
                                      (70, 200, 255), CAR_LINE_THICKNESS)
                        on_queue.append(((x+onep*(i),y,onep,h),(int(x + onep*(i+1) / 2), int(y + h / 2))))
                        ct = (int(x + onep*(i+1) / 2), int(y + h / 2))
                        cv2.circle(frame, ct, 2, (70, 200, 255), -1)
                    cv2.rectangle(frame, (x + onep * (int(w / onep) - 1), y), (x + w - 1, y + h - 1),
                                  (70, 200, 255), CAR_LINE_THICKNESS)
            else:
                cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1),
                              (0, 0, 255), CAR_LINE_THICKNESS)
                # 자동차 이동 추적하는 원
                cv2.circle(frame, centroid, 2, (0, 0, 255), -1)
                off_queue.append(match)

        if frame_no % 20 == 1:
            print("Frame.no: " + str(frame_no), "|", "Total object: " + str(len(blobs)), "|", "on_Queue: " + str(
                len(on_queue)), "|", "off_Queue: " + str(len(off_queue)))
            o_q_cnt = len(on_queue)
            
        cv2.putText(frame, ("count: %s" % str(o_q_cnt)), (570, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        
        if car_counter is None:
            print("Creating vehicle counter...")
            car_counter = VehicleCounter(
                frame.shape[:2], 7*frame.shape[0] / 10)
#            car_counter = VehicleCounter(frame.shape[:2], 3*frame.shape[0] / 3)

        # cv2.putText(frame, ("count: %03d" % len(on_queue)),
        #                 (12, 56), cv2.FONT_HERSHEY_PLAIN, 1.2, (127, 255, 255), 2)
        # get latest count
        car_counter.update_count(blobs, frame)
        # vehicle_RHS: VehicleCount 클래스의 오른쪽 부분에 vehicle이 몇개 있는지 count해주는 변수, update_count에서 증가
        # vehicle_LHS: VehicleCount 클래스의 변수, update_count에서 증가
        current_count = car_counter.vehicle_RHS + car_counter.vehicle_LHS

        # print elapsed time to console
        elapsed_time = time.time()-start_time
#        print("-- %s seconds --" % round(elapsed_time,2))

        # output video
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

#         # draw dividing line
#         # flash green when new car counted, 횡단보도 앞 선 표
#         if current_count > total_cars:
#             cv2.line(frame, (0, int(10.6*frame_h/10)),(frame_w, int(6.8*frame_h/10)),
#                  (0,255,0), 4*LINE_THICKNESS)
# #            cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
# #                 (0,255,0), 4*LINE_THICKNESS)
#         else:
#             cv2.line(frame, (0, int(10.6*frame_h/10)),(frame_w, int(6.8*frame_h/10)),
#              (0,0,255), 2*LINE_THICKNESS)
# #            cv2.line(frame, (0, int(2*frame_h/3)),(frame_w, int(2*frame_h/3)),
# #             (0,0,255), 2*LINE_THICKNESS)
#
        # update with latest count
        total_cars = current_count

        # draw upper limit
        # cv2.line(frame, (0, 100),(frame_w, 100), (0,0,0), LINE_THICKNESS)

        cv2.imshow("preview", frame)
        out.write(frame)

        if cv2.waitKey(1) == 27:
            break
    else:
        break

# cv2.line()
cv2.destroyAllWindows()
cap.release()
out.release()
