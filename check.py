#data 2022.4.22

import numpy as np

event_info = np.load('event_info.npy')
event_info = event_info.tolist()
for i in event_info:
    print(i)