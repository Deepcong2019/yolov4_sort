# -*- coding: utf-8 -*-
"""
Time    : 2022/5/12 08:35
Author  : cong
"""
from sort_class import *
from PIL import Image
from yolo_slim import *
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
video_path = "video/310_r11.mp4"
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
min_hits = 1
iou_threshold = 0.3
trackers = []
frame_count = 0

while ref:
    ref, frame = capture.read()
    if not ref:
        break
    count_id += 1
    print('count_id:', count_id)
    # -------------------------------------------------------------------------#
    # 图像进行处理并检测
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    hatch, cargo = yolo.detect_image(frame)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    frame_count += 1

    # 每一帧输出结果，开始跟踪过程
    # get predicted locations from existing trackers.
    if cargo:
        dets = np.array(cargo)
        # get predicted locations from existing trackers.
        trks = np.zeros((len(trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            cv2.rectangle(frame, (int(pos[1])-2, int(pos[0])-2), (int(pos[3])+2, int(pos[2])+2), (0, 255, 0), 2)
            cv2.putText(frame, "Green: kalman_predict_box, slightly enlarge ", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Red: yolov4_predict_box, normal size box ", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            trackers.pop(t)
        # associate_detections_to_trackers:详细过程见 https://blog.csdn.net/DeepCBW/article/details/124740092
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, iou_threshold)
        print('unmatched_dets:', unmatched_dets)
        # update matched trackers with assigned detections
        # 只有匹配上才会update，此时，每个track的time_since_update重新归置为0，
        # 没有匹配上轨迹的detections，将会赋予新的轨迹。
        # 新轨迹连续匹配上update时，hit_streak += 1，大于min_hits时才会赋予新的id。
        # 新轨迹未如果没匹配上detection，这个track的time_since_update+=1，重新将hit_streak = 0，直到这个trk.time_since_update>max_age,删除该轨迹。
        for m in matched:
            trackers[m[1]].update(dets[m[0], :])
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            trackers.append(trk)
        i = len(trackers)
        for trk in reversed(trackers):
            d = trk.get_state()[0]
            print('trk.hit_streak:', trk.hit_streak)
            if (trk.time_since_update < 1) and (trk.hit_streak >= min_hits or frame_count <= min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > max_age:
                trackers.pop(i)
        if len(ret) > 0:
            trajectories = np.concatenate(ret)
        else:
            trajectories = np.empty((0, 5))
        # 根据跟踪结果，绘图
        for i in trajectories:
            track_id = i[4]
            print('track_id:', track_id)
            x1, y1, x2, y2 = int(i[1]), int(i[0]), int(i[3]), int(i[2])
            print('box:', x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "track_id: %d " % track_id, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "frame_id: %d " % count_id, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
print("Video Detection Done!")
capture.release()
print("Save processed video to the path:", video_save_path)
out.release()
cv2.destroyAllWindows()
yolo.close_session()
