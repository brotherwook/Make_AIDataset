import cv2
import os

# %% 동영상 이름으로 로드/ 이미지 캡쳐 후 저장할 경로 생성
path = 'C:/MyWorkspace/datasets/'
video_name = "F20010_5_202010061530"
video_path = os.path.join(path, "video", video_name + ".avi")

video = cv2.VideoCapture(video_path)
fps = round(video.get(cv2.CAP_PROP_FPS))

try:  # 비디오 이름으로 저장경로 생성
    os.mkdir(os.path.join(path, "results", video_name))
    os.mkdir(os.path.join(path, "results", video_name, "images"))
    os.mkdir(os.path.join(path, "results", video_name, "boxed"))
except:  # 생성할 경로가 이미 존재할경우 에러 생기는데 이 경우 패스
    pass
# %% fps 측정해서 캡쳐 ##
cnt = 0
num = 1
while True:
    rtv, frame = video.read()
    if not rtv:
        break
    if cnt % fps == 0:
        cv2.imwrite(os.path.join(path, "train", video_name, 'images', video_name + '_{0:04d}.jpg'.format(num)), frame)
        num += 1
    cnt += 1
