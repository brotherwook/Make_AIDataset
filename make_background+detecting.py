import cv2
import numpy as np
import time
import sys
import os
# video_path = 'D:/video/20201008/F20003_4_202010080900_ROI.avi'
# video_path = "C:/Users/user/Desktop/영상/인코딩/F20003_4_202011020900.avi"

def main(video_path, save_path):

    img_name = video_path[-25:-4]

    # 디렉토리 생성
    try:
        os.mkdir(save_path + "/" + img_name)
    except:
        print(img_name, "폴더가 이미 있음")
    try:
        os.mkdir(save_path + "/" + img_name + "/original")
    except:
        print(img_name + "/original 폴더가 이미있음")
    try:
        os.mkdir(save_path + "/" + img_name + "/roi")
    except:
        print(img_name + "/roi 폴더가 이미 있음")
    try:
        os.mkdir(save_path + "/" + img_name + "/detect")
    except:
        print(img_name + "/detect 폴더가 이미 있음")

    start = time.time()
    img_name = video_path[-25:-4]
    # 비디오 불러오기
    cap = cv2.VideoCapture(video_path)
    if (not cap.isOpened()):
        print('Error opening video')

    fps = cap.get(cv2.CAP_PROP_FPS)
    height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    acc_gray = np.zeros(shape=(int(height/2), int(width/2)), dtype=np.float32)
    acc_bgr = np.zeros(shape=(int(height/2), int(width/2), 3), dtype=np.float32)
    t = 0

    for i in range(int(fps*60)):
        _, frame = cap.read()
        frame = cv2.resize(frame, (int(width / 2), int(height / 2)))
        cv2.accumulate(frame, acc_bgr)
        avg_bgr = acc_bgr / i
        dst_bgr = cv2.convertScaleAbs(avg_bgr)
        print("\r 배경생성:", int(i/(fps*60)*100), "%", end="")
        # cv2.imshow("dst_bgr",dst_bgr)
        cv2.waitKey(1)
        t = i
    cap.release()

    cap = cv2.VideoCapture(video_path)
    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE
    dets = []

    # 동영상 저장용 (안씀)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
    out = cv2.VideoWriter(save_path + '/test2.avi', fourcc, 30.0, (int(width/2), int(height/2)))

    while True:
        try:
            retval, frame = cap.read()
            if not retval:
                break
            frame = cv2.resize(frame, (int(width/2), int(height/2)))
            t += 1

            cv2.accumulate(frame, acc_bgr)
            avg_bgr = acc_bgr / t
            dst_bgr = cv2.convertScaleAbs(avg_bgr)

            # cv2.imshow("dst_bgr", dst_bgr)

            # if t % 30 == 0:
            diff_bgr = cv2.absdiff(frame, dst_bgr)

            db, dg, dr = cv2.split(diff_bgr)
            ret, bb = cv2.threshold(db, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret, bg = cv2.threshold(dg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ret, br = cv2.threshold(dr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow("bb", bb)
            # cv2.imshow("bg", bg)
            # cv2.imshow("br", br)

            bImage = cv2.bitwise_or(bb, bg)
            bImage = cv2.bitwise_or(br, bImage)

            median = cv2.medianBlur(bImage, 5)

            dil = cv2.dilate(median, None, 10)

            contours, hierarchy = cv2.findContours(dil, mode, method)

            dets = []
            # 중심점 구하기
            for i, contour in enumerate(contours):
                dil = cv2.fillPoly(dil, contour, 255)
                x, y, w, h = cv2.boundingRect(contour)
                if 20 < w < 350 and 20 < h < 200:
                    dets.append(np.array([y, x, y + h, x + w]))
            # cv2.imshow("dil", dil)

            for det in dets:
                frame = cv2.rectangle(frame, (det[1], det[0]), (det[3],det[2]), (255,0,0), 2, cv2.LINE_AA)

            out.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == 27:
                break
            print("\r 영상재생:",(t-1800)/(fps*360)*100,"%", end="")

        except KeyboardInterrupt:
            break



    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
    cap.release()
    out.release()

    end = time.time()
    print("\n", int((end - start)/60),"분")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        video_path = './inputs/F20003_4_202011021900.avi'
        save_path = './outputs'
    elif len(sys.argv) == 3:
        video_path = sys.argv[1]
        save_path = sys.argv[2]
    else:
        print("Usage: python person_detecting.py input_video_path(avi) output_image_dir output_image_name")
        print("원본 동영상 경로 및 저장할 배경이미지 경로(파일이름까지)(경로에 한글 포함되면 안됩니다.)")

    main(video_path, save_path)