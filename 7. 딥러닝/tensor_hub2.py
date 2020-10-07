import numpy as np
import cv2
import tensorflow_hub as hub
import os
import xml.etree.ElementTree as et

# %%  모델 로드 및 xml 저장경로 설정 /캡쳐된 이미지 경로 설정
model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_800x1333/1")

video_name = "F20003_7_202010061530"

path = os.path.join("C:/MyWorkspace/datasets/results", video_name)
xml_path = os.path.join(path, video_name + "_coding.xml")
images_path = os.path.join(path, "images")
images = os.listdir(images_path)
root = et.Element("annotations")
num = 0

# %%  모델로 측정 후 프레임당 한번씩 xml 덮어쓰기/ 박싱된 이미지 저장(원본 이미지와 이름은 같고 저장경로가 다름)
for image in images:
    img = cv2.imread(os.path.join(images_path, image))

    height, width = img.shape[:2]
    tag = et.SubElement(root, "image", {"id": str(num), "name": image, "width": str(width), "height": str(height)})

    input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = input[np.newaxis, ...]
    output = model(input)

    threshold = 0.2

    for i, score in enumerate(output["detection_scores"][0]):
        if score < threshold:
            break
        ymin, xmin, ymax, xmax = output["detection_boxes"][0][i]
        (left, top) = (int(xmin * width), int(ymin * height))
        (right, bottom) = (int(xmax * width), int(ymax * height))
        if (right - left) * (bottom - top) > 10000:  # 너무 클 경우 무시
            continue  # 영상마다 측정하여 수치변경
        # if (right - left) * (bottom - top) < 1100:  # 너무 작은것도 무시
        #     continue
        if output["detection_classes"][0][i] == 1:  # <= 8: # 1~8 = 사람,자전거,승용차,오토바이,비행기,버스,전철,트럭
            # 참고: https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt
            img = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 2)
            # class_entity = "{:.2f}%".format(score * 100)
            # cv2.putText(img_original, class_entity, (left, bottom + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4,
            #             (0, 0, 0), 1)
            et.SubElement(tag, "box", {"label": "사람", "occluded": "0", "source": "manual",
                                       "xbr": str(right), "xtl": str(left),
                                       "ybr": str(bottom), "ytl": str(top)})
    cv2.imwrite(os.path.join(path, "boxed", image), img)
    xml = et.ElementTree(root)
    xml.write(xml_path)
    num += 1
    # cv2.imshow("", img)
    # if cv2.waitKey(1) == 27:
    #     break
    if num % 4 == 0:
        loading = "\\"
    elif num % 4 == 1:
        loading = "|"
    elif num % 4 == 2:
        loading = "/"
    else:
        loading = "-"
    print("\r"+loading +"진행중 {:.2f}%".format((num / len(images)) * 100), end='')
# cv2.destroyAllWindows()
print("\r완료                        ")
