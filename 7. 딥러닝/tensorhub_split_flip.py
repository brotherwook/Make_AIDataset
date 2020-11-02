import numpy as np
import cv2
import tensorflow_hub as hub
import os
import xml.etree.ElementTree as et

# %%
model = hub.load("https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_1024x1024/1")

num = 2

path = "C:/MyWorkspace/datasets/train"
xml_path = os.path.join(path, "annotations.xml")
images_path = os.path.join(path, "images")
images = os.listdir(images_path)
xml = et.parse(xml_path)
root = xml.getroot()

# %%
for image in images:
    img = cv2.imread(os.path.join(images_path, image))

    height, width = img.shape[:2]
    half = int(width / 2)
    qut = int(half / 2)

    for c in range(3):
        if c == 0:
            input = img[:, :half]
        elif c == 1:
            input = img[:, half:]
        else:
            input = img[:, qut:width - qut]
        input = cv2.flip(input, 1)
        input = np.swapaxes(input, 0, 1)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = input[np.newaxis, ...]
        output = model(input)
        # 바인딩 박스 드로잉
        threshold = 0.2
        for i, score in enumerate(output["detection_scores"][0]):
            if score < threshold:
                break
            xmax, ymin, xmin, ymax = output["detection_boxes"][0][i]
            (left, top) = (int((1 - xmin) * half), int(ymin * height))
            (right, bottom) = (int((1 - xmax) * half), int(ymax * height))
            if c == 0:
                if (left + right) / 2 > (half * 0.75):
                    continue
            elif c == 1:
                if (left + right) / 2 < (half * 0.25):
                    continue
                else:
                    left, right = left + half, right + half
            else:
                if (left + right) / 2 > (half * 0.75) or (left + right) / 2 < (half * 0.25):
                    continue
                else:
                    left, right = left + qut, right + qut
            if output["detection_classes"][0][i] == 1:  # <= 8: # 1~8 = 사람,자전거,승용차,오토바이,비행기,버스,전철,트럭
                # 참고: https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt
                if (right - left) * (bottom - top) > 10000:  # 너무 클 경우 무시 #영상마다 측정하여 수치변경
                    continue
                if (right - left) * (bottom - top) < 900:  # 너무 작은것도 무시
                    continue
                if bottom < height * 0.5:  # 인도만
                    continue
                img_original = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 1)
                # class_entity = "{:.2f}%".format(score * 100)
                # cv2.putText(img, class_entity, (left, bottom + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.3,
                #             (255, 255, 255), 1)

                et.SubElement(root[num], "box", {"label": "사람", "occluded": "0", "source": "manual",
                                                 "xbr": str(right), "xtl": str(left),
                                                 "ybr": str(bottom), "ytl": str(top)})
    xml.write(xml_path)
    num += 1
    cv2.imshow('', img)
    if cv2.waitKey(1) == 27:
        break
    print("\r진행상황: {:.2f}%".format(((num - 2) / len(images)) * 100), end='')
cv2.destroyAllWindows()
