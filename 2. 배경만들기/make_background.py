"""
배경화면 생성기 입니다
동영상중 자동차 정차구간이 많은 영상에는 적합하지 않습니다
자동차가 영상 내내 주차/정차 되어 있을 경우 그 자동차도 배경으로 인식합니다
빨간불이 긴 영상에서는 편집기로 해당부분을 자른 후 사용하시면 될 것 같습니다
"""

import cv2
import numpy as np
import random

# %% 원본 동영상 경로 및 저장할 (배경)이미지 경로(파일이름까지)
video_path = "C:/MyWorkspace/Detectioncode/inputs/back_sample2.mp4"
# %% 원본 동영상 읽고 정보 추출
cctv = cv2.VideoCapture(video_path)

frame = cctv.get(cv2.CAP_PROP_FRAME_COUNT)  # 프레임 수
width = cctv.get(cv2.CAP_PROP_FRAME_WIDTH)  # 너비
height = cctv.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 높이
# %% 동영상을 nparray로 변환하기위한 껍데기 제작 // 모든 bgr 값이 0,0,0 인 동영상 어레이 생성
video_np = np.zeros(shape=(int(frame), int(height), int(width), 3), dtype=np.int16)
video_np1 = np.zeros(shape=(int(240), int(height), int(width), 3), dtype=np.int16)
video_np2 = np.zeros(shape=(int(frame-240), int(height), int(width), 3), dtype=np.int16)

# %% 저장할 이미지 껍데기 생성 // 모든 bgr 값이 0,0,0 인 이미지 어레이 생성
background_img = np.zeros(shape=(int(height), int(width), 3), dtype=np.int16)
# %% 동영상 정보를 껍데기에 입력
i = 0
cnt1 = 0
cnt2 = 0
flag = 0

while True:
    ret, image = cctv.read()
    if not ret:
        break

    video_np[i] = image


    if i < 240:
        video_np1[cnt1] = image
        cnt1 += 1
    else:
        video_np2[cnt2] = image
        cnt2 += 1

    i += 1

    if flag == 0:
        roi1 = np.array([[
            (51, 1),
            (548, 479),
            (729, 478),
            (435, 216),
            (158, 1)
        ]], dtype=np.int32)

        roi2 = np.array([[
            (0, 1),
            (2, 101),
            (341, 479),
            (554, 478),
            (56, 0)
        ]], dtype=np.int32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # mask 생성, 마스크를 적용하여 ROI를 제외한 나머지 부분을 0(검은색)으로 만들기 위해
        mask1 = np.zeros(image.shape)  # 이미지와 같은 크기의 마스크 생성
        mask2 = np.zeros(image.shape)  # 이미지와 같은 크기의 마스크 생성

        # 마스크 적용
        # 마스크(원본과 같은 사이즈의 크기)에서 roi영역만 255로 채움
        cv2.fillPoly(mask1, roi1, (255, 255, 255))
        cv2.fillPoly(mask2, roi2, (255, 255, 255))

        nonzero1 = np.nonzero(mask1)
        nonzero2 = np.nonzero(mask2)

        cv2.imshow("mask1", mask1)
        cv2.imshow("mask2", mask2)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            flag = 1

# %% 전체중 랜점한 500개 이미지만 사용(속도향상)
if frame > 500:
    random.shuffle(video_np)  # 여기가 오래 걸립니다 반응이 없다면 섞는 중 입니다.
    video_np = video_np[:500, ...]
# %% 동영상 각 픽셀 위치에서 가장 많이 나온 bgr 값을 배경으로 인식, 이미지 픽셀별로 저장
count = 1
a = width * height
for j in range(int(height)):
    for k in range(int(width)):
        pix = video_np[:, j, k]
        val, cnt = np.unique(pix, return_counts=True, axis=0)
        max = np.argmax(cnt)
        bgr = val[max]
        background_img[j, k] = bgr
        print("진행상황: {:.2f}%".format((count / a) * 100))
        count += 1
# %%
save_path = "C:/MyWorkspace/Detectioncode/temp/1.png"
cv2.imwrite(save_path, background_img)

#%%
count = 1
a = len(nonzero1[0])

for x, y in zip(nonzero1[0], nonzero1[1]):
    pix = video_np1[:, x, y]
    val, cnt = np.unique(pix, return_counts=True, axis=0)
    max = np.argmax(cnt)
    bgr = val[max]
    background_img[x, y] = bgr
    print("진행상황: {:.2f}%".format((count / a) * 100))
    count += 1

# %%
save_path = "C:/MyWorkspace/Detectioncode/temp/2.png"
cv2.imwrite(save_path, background_img)

#%%
count = 1
a = len(nonzero2[0])

for x, y in zip(nonzero2[0], nonzero2[1]):
    pix = video_np2[:, x, y]
    val, cnt = np.unique(pix, return_counts=True, axis=0)
    max = np.argmax(cnt)
    bgr = val[max]
    background_img[x, y] = bgr
    print("진행상황: {:.2f}%".format((count / a) * 100))
    count += 1

# %%
save_path = "C:/MyWorkspace/Detectioncode/temp/3.png"
cv2.imwrite(save_path, background_img)