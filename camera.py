# -*- coding: utf-8 -*-
"""
Time    : 2022/5/13 17:20
Author  : cong
"""
import time

import cv2
count =0
cap = cv2.VideoCapture(0)


while True:
    count += 1
    print('count', count)
    ret, frame = cap.read()
    cv2.putText(frame, "frame_id: %d " % count, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    time.sleep(0.2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

