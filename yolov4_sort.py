# -*- coding: utf-8 -*-
"""
Time    : 2022/5/11 16:15
Author  : cong
"""
from sort_class import *
from PIL import Image
from yolo_slim import *
import copy, math, cv2, datetime
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import re
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
yolo = YOLO()

# -------------------------------------------------------------------------#
# local
video_path = "video/111111.mp4"
video_save_path = re.split('.mp4', video_path)[0] + '_result.mp4'

# -------------------------------------------------------------------------#

video_fps = 25
capture = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
total_frame = capture.get(7)
print('total_frame:', total_frame)
ref, frame = capture.read()
if not ref:
    raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
count_id = 0

max_age = 100
min_hits = 3
iou_threshold = 0.3
mot_tracker = Sort(max_age, min_hits, iou_threshold)  # create instance of the SORT tracker

while ref:
    t_o = time.time()
    ref, frame = capture.read()
    if not ref:
        break
    count_id += 1
    print('count_id:', count_id)
    # -------------------------------------------------------------------------#
    #图像进行处理并检测
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    hatch, cargo = yolo.detect_image(frame)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    if cargo:
        trackers = mot_tracker.update(np.array(cargo))
        for i in trackers:
            track_id = i[4]
            print('track_id:', track_id)
            x1, y1, x2, y2 = int(i[1]), int(i[0]), int(i[3]), int(i[2])
            print('box:', x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "track_id: %d " % track_id, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "frame_id: %d " % (count_id),  (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Video Detection Done!")
capture.release()
print("Save processed video to the path :", video_save_path)
out.release()
cv2.destroyAllWindows()
yolo.close_session()