import cv2
import numpy as np
import sys
import os
import time
from xml.etree.ElementTree import Element, SubElement, ElementTree

# %%
day = ['1145', '1200', '1215', '1230', '1245', '1300', '1315']
img_name = None
a=0

def imgprocessing(img, roi, dst_bgr, name, id, flag):
    global img_name
    global day
    global a

    mode = cv2.RETR_EXTERNAL
    method = cv2.CHAIN_APPROX_SIMPLE

    temp = cv2.resize(img, None, fx=0.5, fy=0.5)
    diff_bgr = cv2.absdiff(temp, dst_bgr)

    db, dg, dr = cv2.split(diff_bgr)
    ret, bb = cv2.threshold(db, 50, 255, cv2.THRESH_BINARY)
    ret, bg = cv2.threshold(dg, 50, 255, cv2.THRESH_BINARY)
    ret, br = cv2.threshold(dr, 50, 255, cv2.THRESH_BINARY)
    bb = cv2.blur(bb, (12,2))
    bg = cv2.blur(bg, (12,2))
    br = cv2.blur(br, (12,2))

    bImage = cv2.bitwise_or(bb, bg)
    bImage = cv2.bitwise_or(br, bImage)

    median = cv2.medianBlur(bImage, 5)

    dil = cv2.dilate(median, None, 10)
    if flag == 0:
        dil = cv2.line(dil, (186,409), (1243,392), 0, 3, cv2.LINE_AA)
    else:
        dil = cv2.line(dil, (186,69), (1243,51), 0, 3, cv2.LINE_AA)
    contours, hierarchy = cv2.findContours(dil, mode, method)

    dets = []
    # 중심점 구하기
    for i, contour in enumerate(contours):
        dil = cv2.fillPoly(dil, contour, 255)
        x, y, w, h = cv2.boundingRect(contour)
        if 30 < w < 400 and 20 < h < 100:
            if isPointInPolygon(x, y, w, h, roi, flag):
                dets.append(np.array([y, x, y + h, x + w]))
                bbox = (x, y, x + w, y + h)
                # kh, kw = h, w
                if w < 80:
                    label = "이륜차"
                    color = (255, 0, 0)
                elif w > 180:
                    label = "버스"
                    color = (0, 255, 0)
                else:
                    label = "승용/승합"
                    color = (0, 255, 255)

                img = cv2.rectangle(img, (x*2, y*2), ((x + w) * 2, (y + h) * 2), color, 2, cv2.LINE_AA)

                if flag == 0:
                    makeXML(id, name, bbox, label, 0)
                else:
                    makeXML(id, name, bbox, label, 1)


    return img, dil

# %%
def isPointInPolygon(x, y, w, h, roi, flag):
    x2 = x+w
    y2 = y+h
    # 한점이라도 roi안에 있으면 인식
    # up frame
    if flag == 0:
        if roi[0][0] < x < roi[3][0] and roi[3][1] +10 < y < roi[2][1]-25:
            return True
        if roi[0][0] < x < roi[3][0] and roi[3][1] +10 < y2 < roi[2][1]-25:
            return True
        if roi[0][0] < x2 < roi[3][0] and roi[3][1] +10 < y2 < roi[2][1]-25:
            return True
        if roi[0][0] < x2 < roi[3][0] and roi[3][1] +10 < y < roi[2][1]-25:
            return True
    # down frame
    else:
        if roi[0][0] < x < roi[3][0] and roi[0][1] +20 < y < roi[1][1]-10:
            return True
        if roi[0][0] < x < roi[3][0] and roi[0][1] +20 < y2 < roi[1][1]-10:
            return True
        if roi[0][0] < x2 < roi[3][0] and roi[0][1] +20 < y2 < roi[1][1]-10:
            return True
        if roi[0][0] < x2 < roi[3][0] and roi[0][1] +20 < y < roi[1][1]-10:
            return True

    return False



# %% xml 만드는 부분
root_up = Element('annotations')
root_down = Element('annotations')
width = 0
height = 0


def makeXML(id, name, bbox, label, flag):
    global root_up
    global root_down

    if flag == 0:
        image = root_up.find('image[@id="' + str(id) + '"]')
        if image is None:
            image = SubElement(root_up, 'image')
            image.attrib["id"] = str(id)
            image.attrib["name"] = name
            image.attrib["width"] = str(width)
            image.attrib["height"] = str(height)
    else:
        image = root_down.find('image[@id="' + str(id) + '"]')
        if image is None:
            image = SubElement(root_down, 'image')
            image.attrib["id"] = str(id)
            image.attrib["name"] = name
            image.attrib["width"] = str(width)
            image.attrib["height"] = str(height)

    box = SubElement(image, 'box')
    box.attrib["label"] = label
    box.attrib["occluded"] = '0'
    box.attrib["source"] = 'manual'

    xtl = bbox[0] * 2
    ytl = bbox[1] * 2
    xbr = bbox[2] * 2
    ybr = bbox[3] * 2

    if xtl < 0:
        xtl = 0
    if ytl < 0:
        ytl = 0
    if xbr > 2592:
        xbr = 2592
    if ybr > 1944:
        ybr = 1944

    box.attrib["xtl"] = str(xtl)
    box.attrib["ytl"] = str(ytl)
    box.attrib["xbr"] = str(xbr)
    box.attrib["ybr"] = str(ybr)


# 들여쓰기 함수
def apply_indent(elem, level=0):
    # tab = space * 2
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for elem in elem:
            apply_indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def number(num):
    return '%05d' % num


# %%
def main(video_path, save_path, file_name=None):
    global width
    global height
    global root_down
    global root_up
    global img_name

    start = time.time()

    if file_name is None:
        start = time.time()
        file_name = [video_path[-25:]]
        video_path = video_path[:-25]
        repeat = 1
    else:
        repeat = len(file_name)

    for re in range(repeat):
        path = os.path.join(video_path, file_name[re])
        img_name = path[-25:-4]

        print(path)
        print(img_name)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        # 디렉토리 생성
        try:
            os.mkdir(save_path, mode=0o777)
        except:
            print(save_path, "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_up", mode=0o777)
        except:
            print(save_path + "/" + "_up", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_down", mode=0o777)
        except:
            print(save_path + "/" + "_down", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_up/detect", mode=0o777)
        except:
            print(save_path + "/" + "_up/detect", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_down/detect", mode=0o777)
        except:
            print(save_path + "/" + "_down/detect", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_up/roi", mode=0o777)
        except:
            print(save_path + "/" + "_up/roi", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_down/roi", mode=0o777)
        except:
            print(save_path + "/" + "_down/roi", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_up/xml", mode=0o777)
        except:
            print(save_path + "/" + "_up/xml", "폴더가 이미 있음")
        try:
            os.mkdir(save_path + "/" + "_down/xml", mode=0o777)
        except:
            print(save_path + "/" + "_down/xml", "폴더가 이미 있음")

        # 배경틀 만들기
        acc_bgr = np.zeros(shape=(int(height / 2), int(width / 2), 3), dtype=np.float32)
        t = 0
        up = None
        down = None

        # 60초정도의 프레임으로 배경을 미리 만들어놓음 (초반에 튀는값을 막기위해서)
        for i in range(1, int(fps*60), 1):
            ret, frame = cap.read()
            if not ret:
                print("video error")
            resize_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            cv2.accumulate(resize_frame, acc_bgr)
            avg_bgr = acc_bgr / i
            dst_bgr = cv2.convertScaleAbs(avg_bgr)
            print("\r 배경생성:", int(i/(fps*60)*100), "%", end="")
            # 나중에 t=0으로 놔두면 0으로 나누게 되서 에러가 남
            t = i
        print()

        # 저장할 동영상의 shape을 얻기위해
        up = frame[72 * 2:536 * 2, :].copy()
        down = frame[414 * 2:920 * 2, :].copy()

        # 영상을 다시 처음부터 재생하기 위해
        cap.release()
        cap = cv2.VideoCapture(path)
        print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # # 동영상 저장용
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 디지털 미디어 포맷 코드 생성 , 인코딩 방식 설
        out1 = cv2.VideoWriter(save_path + "/" + "_up/roi/" + img_name + "_up.avi", fourcc, int(fps), (up.shape[1], up.shape[0]))
        out2 = cv2.VideoWriter(save_path + "/" + "_down/roi/" + img_name + "_down.avi", fourcc, int(fps), (down.shape[1], down.shape[0]))
        out3 = cv2.VideoWriter(save_path + "/" + "_up/detect/" + img_name + "_up.avi", fourcc, 10,
                               (up.shape[1], up.shape[0]))
        out4 = cv2.VideoWriter(save_path + "/" + "_down/detect/" + img_name + "_down.avi", fourcc, 10,
                               (down.shape[1], down.shape[0]))

        # 위쪽차선, 아래쪽차선 각각의 ROI영역설정
        up_roi = np.array([[420, 80], [420, 405], [1020, 378], [1020, 118]])
        down_roi = np.array([[420, 70], [420, 430], [1020, 445], [1020, 52]])

        # 실제 프레임수 초기화
        cnt = 0
        # xml저장을 위한 id 초기화
        id = 0

        # xml초기화 후 다시 annotations 만듬
        root_up = Element('annotations')
        root_down = Element('annotations')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 영상 크롭
            up = frame[72 * 2:536 * 2, :].copy()
            down = frame[414 * 2:920 * 2, :].copy()

            # 리사이즈해서 배경만들기
            img = cv2.resize(frame, None, fx=0.5, fy=0.5)
            t += 1
            cv2.accumulate(img, acc_bgr)
            avg_bgr = acc_bgr / t
            dst_bgr = cv2.convertScaleAbs(avg_bgr)

            # 만든 배경 크롭
            up_bgr = dst_bgr[72:536, :]
            down_bgr = dst_bgr[414:920, :]

            # xml에 들어갈 이미지 이름
            name = "/" + img_name + "_" + number(cnt) + ".jpg"
            print("\r", cnt, end="")

            if cnt % int(fps) == 0:
                up_clone = up.copy()
                down_clone = down.copy()
                cv2.polylines(up_clone, [up_roi * 2], True, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.polylines(down_clone, [down_roi * 2], True, (0, 0, 255), 2, cv2.LINE_AA)

                up, up_dil = imgprocessing(up, up_roi, up_bgr, name, id, 0)
                down, down_dil = imgprocessing(down, down_roi, down_bgr, name, id, 1)

                # ----------------------------------------------

                cv2.polylines(up, [up_roi * 2], True, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.polylines(down, [down_roi * 2], True, (0, 0, 255), 2, cv2.LINE_AA)

                # 영상으로 저장
                out1.write(up_clone)
                out2.write(down_clone)
                out3.write(up)
                out4.write(down)

                # dectecting 결과를 확인하면서 파일을 생성하려면 주석 해제
                # cv2.imshow("up", up)
                # cv2.imshow("dwon", down)
                # key = cv2.waitKey(1)
                # if key == 27:
                #     break

                # 저장되는 xml의 id번호
                id += 1

            # 현재 프레임 번호
            cnt += 1


        cv2.destroyAllWindows()
        cap.release()
        out1.release()
        out2.release()
        out3.release()
        out4.release()

        end = time.time()
        print("\n", int((end - start) / 60), "분")

        # 들여쓰기
        apply_indent(root_up)
        apply_indent(root_down)

        # xml 파일로 보내기
        tree = ElementTree(root_up)
        tree.write(save_path + "/" + "_up/xml/" + img_name + "_up.xml")
        tree = ElementTree(root_down)
        tree.write(save_path + "/" + "_down/xml/" + img_name + "_down.xml")
        
        # xml 다 지우기
        root_down.clear()
        root_up.clear()
        root_down = None
        root_up = None

if __name__ == '__main__':
    os.chdir("./")

    if len(sys.argv) == 1:
        video_path = './input/F20003_4_202010231315.avi'
        save_path = "./crop"
        main(video_path, save_path)
    if len(sys.argv) == 3:
        file_path = sys.argv[1]
        save_path = sys.argv[2]
        file_name = os.listdir(file_path)
        main(file_path, save_path, file_name)
    else:
        print("Usage: python car_detection_F20003_4.py input_directory_path output_dir_path")
        print("원본 동영상 경로 및 저장할 배경이미지 경로(파일이름까지)(경로에 한글 포함되면 안됩니다.)")