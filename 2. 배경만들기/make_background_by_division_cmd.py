import cv2
import numpy as np
import sys
import os


class Division_frame():
    def __init__(self, frame, num):
        self.num = num
        self.frame = frame
        self.stop = True
        self.acc_gray = np.zeros(shape=(frame.shape[0], frame.shape[1]), dtype=np.float32)
        self.acc_bgr = np.zeros(shape=(frame.shape[0], frame.shape[1], 3), dtype=np.float32)
        self.t = 1
        self.dst_bg = None
        self.dst_gray = None

    def change_frame(self, frame):
        self.frame = frame

    def background_start(self):

        cv2.accumulate(self.frame, self.acc_bgr)
        self.avg_bgr = self.acc_bgr / self.t
        self.dst_bgr = cv2.convertScaleAbs(self.avg_bgr)

        # 흑백 배경이미지 필요할시 사용
        # gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # cv2.accumulate(gray, self.acc_gray)
        # self.avg_gray = self.acc_gray / self.t
        # self.dst_gray = cv2.convertScaleAbs(self.avg_gray)

        self.t += 1

        return self.dst_gray, self.dst_bgr

    def change_stop(self):
        if self.stop is True:
            # print(self.num, ": False")
            self.stop = False
        else:
            # print(self.num, ": True")
            self.stop = True

        return self.stop


# %% 마우스 클릭을 위한 부분
clicked_points = []
clone = None

# MouseLeftClick 이벤트 핸들러에서 사용하는 전역변수
midline = 0
firstline = 0
secondline = 0
frame1, frame2, frame3, frame4, frame5, frame6 = None, None, None, None, None, None
flag = [False for i in range(6)]
total = []


def MouseLeftClick(event, x, y, flags, param):
    global flag
    global frame1, frame2, frame3, frame4, frame5, frame6
    global clone
    global midline, firstline, secondline
    global total

    f, s, m = int(firstline / 2), int(secondline / 2), int(midline / 2)
    # 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        if x <= f and y <= m:
            temp = clone[:midline, :firstline, :]
            if frame1 is None:
                frame1 = Division_frame(temp, 0)
                total.append(frame1)
            else:
                frame1.change_frame(temp)
            stop = frame1.change_stop()

            if stop:
                flag[0] = False
            else:
                flag[0] = True

        elif f < x <= s and y <= m:
            temp = clone[:midline, firstline:secondline, :]
            if frame2 is None:
                frame2 = Division_frame(temp, 1)
                total.append(frame2)
            else:
                frame2.change_frame(temp)
            stop = frame2.change_stop()
            if stop:
                flag[1] = False
            else:
                flag[1] = True

        elif s < x and y <= m:
            temp = clone[:midline, secondline:, :]
            if frame3 is None:
                frame3 = Division_frame(temp, 2)
                total.append(frame3)
            else:
                frame3.change_frame(temp)
            stop = frame3.change_stop()
            if stop:
                flag[2] = False
            else:
                flag[2] = True

        elif x <= f and m < y:
            temp = clone[midline:, :firstline, :]
            if frame4 is None:
                frame4 = Division_frame(temp, 3)
                total.append(frame4)
            else:
                frame4.change_frame(temp)
            stop = frame4.change_stop()
            if stop:
                flag[3] = False
            else:
                flag[3] = True

        elif f < x <= s and m < y:
            temp = clone[midline:, firstline:secondline, :]
            if frame5 is None:
                frame5 = Division_frame(temp, 4)
                total.append(frame5)
            else:
                frame5.change_frame(temp)
            stop = frame5.change_stop()
            if stop:
                flag[4] = False
            else:
                flag[4] = True

        elif s < x and m < y:
            temp = clone[midline:, secondline:, :]
            if frame6 is None:
                frame6 = Division_frame(temp, 5)
                total.append(frame6)
            else:
                frame6.change_frame(temp)
            stop = frame6.change_stop()
            if stop:
                flag[5] = False
            else:
                flag[5] = True
        else:
            print("이건 뭘까..")


def bgr_by_division(video_path, save_path, img_name):
    global midline, firstline, secondline
    global clone
    global frame1, frame2, frame3, frame4, frame5, frame6
    global flag
    global total

    # 새 윈도우 창을 만들고 그 윈도우 창에 MouseLeftClick 함수를 세팅해 줍니다.
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseLeftClick)

    # 비디오 불러오기
    cap = cv2.VideoCapture(video_path)
    if (not cap.isOpened()):
        print('Error opening video')

    height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    t = 0

    wait = 0

    midline = int(height / 2)
    firstline = int(width / 3)
    secondline = int(width / 3 * 2)

    background_img = np.zeros(shape=(int(height), int(width), 3), dtype=np.uint8)
    # 흑백 배경이미지 필요할시 사용
    # background_img_gray = np.zeros(shape=(int(height), int(width)), dtype=np.uint8)


    while True:
        retval, frame = cap.read()
        if not retval:
            break
        clone = frame.copy()

        if flag[0]:
            temp = clone[:midline, :firstline, :]
            frame1.change_frame(temp)
            wait = 1
        if flag[1]:
            temp = clone[:midline, firstline:secondline, :]
            frame2.change_frame(temp)
            wait = 1
        if flag[2]:
            temp = clone[:midline, secondline:, :]
            frame3.change_frame(temp)
            wait = 1
        if flag[3]:
            temp = clone[midline:, :firstline, :]
            frame4.change_frame(temp)
            wait = 1
        if flag[4]:
            temp = clone[midline:, firstline:secondline, :]
            frame5.change_frame(temp)
            wait = 1
        if flag[5]:
            temp = clone[midline:, secondline:, :]
            frame6.change_frame(temp)
            wait = 1

        # 영상 분할 라인
        frame = cv2.line(frame, (0, midline), (width, midline), (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.line(frame, (firstline, 0), (firstline, height), (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.line(frame, (secondline, 0), (secondline, height), (255, 255, 255), 2, cv2.LINE_AA)

        for i, v in enumerate(total):
            if not v.stop:
                dst_gray, dst_bgr = v.background_start()
                frame = cv2.rectangle(frame, (int(v.num % 3) * firstline + 5, int(v.num / 3) * midline + 5),
                                      ((int(v.num % 3) + 1) * firstline - 5, (int(v.num / 3) + 1) * midline - 5),
                                      (0, 0, 255), 3, cv2.LINE_AA)
                dst_bgr = cv2.resize(dst_bgr, (int(dst_bgr.shape[1] / 2), int(dst_bgr.shape[0] / 2)))
                cv2.imshow(str(v.num + 1), dst_bgr)
            else:
                cv2.destroyWindow(str(v.num + 1))

            background_img[int(v.num / 3) * midline:(int(v.num / 3) + 1) * midline, int(v.num % 3) * firstline:(int(v.num % 3) + 1) * firstline] = v.dst_bgr
            # 흑백 배경이미지 필요할시 사용
            # background_img_gray[int(v.num / 3) * midline:(int(v.num / 3) + 1) * midline, int(v.num % 3) * firstline:(int(v.num % 3) + 1) * firstline] = v.dst_gray


        back = cv2.resize(background_img,(int(width / 2), int(height / 2)))
        cv2.imshow("background", back)
        # 영상 확인(출력)부분
        frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
        cv2.imshow("image", frame)

        t += 1

        key = cv2.waitKey(wait)
        if key == 32:
            wait = 0

        if key == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

    # for i, v in enumerate(total):
    #     background_img[int(v.num / 3) * midline:(int(v.num / 3)+1) * midline, int(v.num % 3) * firstline:(int(v.num % 3)+1) * firstline ] = v.dst_bgr

    # 마지막 이미지 저장
    # cv2.imwrite(os.path.join(save_path, img_name + '_gray.png'), dst_gray)
    # cv2.imwrite(os.path.join(save_path, img_name + '_bgr.png'), dst_bgr)

    cv2.imwrite(os.path.join(save_path, img_name + '_background.png'), background_img)
    # 흑백 배경이미지 필요할시 사용
    # cv2.imwrite(os.path.join(save_path, img_name + '_background_gray.png'), background_img_gray)

    print("배경이미지 생성 완료")


if __name__ == "__main__":

    if len(sys.argv) == 1:
        video_path = 'D:/F20003_3/20201008/F20003_3_202010080720.avi'
        save_path = "C:/MyWorkspace/Make_AIDataset/backgrounds"
        img_name = "test"
        bgr_by_division(video_path, save_path, img_name)
    elif len(sys.argv) == 4:
        video_path = sys.argv[1]
        save_path = sys.argv[2]
        img_name = sys.argv[3]
        bgr_by_division(video_path, save_path, img_name)
    else:
        print("Usage: python make_background_by_division.py input_video_path(avi) output_image_dir out_image_name")
        print("원본 동영상 저장할 배경이미지 경로 파일이름 (경로에 한글 포함되면 안됩니다.)")

    sys.exit()
