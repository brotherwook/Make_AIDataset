import numpy as np
import cv2
import tensorflow_hub as hub
import time

# %%
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1")

# %%
file_path = "C:/MyWorkspace/datasets/video/person/F18006_2_202009112000.avi"
video = cv2.VideoCapture(file_path)
width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # 너비
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 높이

cnt = 0
blobs = []
while True:
    t1 = time.time()
    retval, img_original = video.read()
    if not retval:
        break
    if cnt % 30 != 0:
        cnt += 1
    else:
        input = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        input = input[np.newaxis, ...]
        output = model(input)

        threshold = 0.2
        i = 0

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
                img_original = cv2.rectangle(img_original, (left, top), (right, bottom), (255, 255, 0), 2)
                center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                blobs.append(center)
                class_entity = "{:.2f}%".format(score * 100)
                cv2.putText(img_original, class_entity, (left, bottom + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                            (255, 255, 255), 1)
            i += 1

        t2 = time.time()
        print(t2 - t1)
        cv2.imshow("", img_original)
        if cv2.waitKey(1) == 27:
            break
        cnt += 1

cv2.destroyAllWindows()
video.release()
