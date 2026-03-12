import pickle
import os
from pathlib import Path
from absl import flags
import sys
import cv2
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("src_dataset",None, "source dataset directory")
flags.DEFINE_string("dst_dataset",None, "path to output dataset")

FLAGS(sys.argv)

demo_dir = Path(FLAGS.src_dataset)
pkls = sorted(demo_dir.glob("*.pkl"))

if not os.path.exists(FLAGS.dst_dataset):
    os.makedirs(FLAGS.dst_dataset)

#hardcoded
successful_demos = ['demonstration_2026-01-18_18:11:48.pkl', 'demonstration_2026-01-18_18:12:15.pkl', 'demonstration_2026-01-18_18:12:47.pkl', 'demonstration_2026-01-18_18:13:12.pkl', 
'demonstration_2026-01-18_18:13:35.pkl', 'demonstration_2026-01-18_18:14:04.pkl', 'demonstration_2026-01-18_18:14:27.pkl', 'demonstration_2026-01-18_18:15:22.pkl', 'demonstration_2026-01-18_18:15:44.pkl',
 'demonstration_2026-01-18_18:16:06.pkl', 'demonstration_2026-01-18_18:16:31.pkl', 'demonstration_2026-01-18_18:16:55.pkl', 'demonstration_2026-01-18_18:17:13.pkl', 'demonstration_2026-01-18_18:17:41.pkl', 
 'demonstration_2026-01-18_18:18:13.pkl', 'demonstration_2026-01-18_18:18:33.pkl', 'demonstration_2026-01-18_18:19:06.pkl', 'demonstration_2026-01-18_18:19:34.pkl', 'demonstration_2026-01-18_18:20:01.pkl', 
 'demonstration_2026-01-18_18:20:20.pkl', 'demonstration_2026-01-18_18:20:48.pkl', 'demonstration_2026-01-18_18:21:21.pkl', 'demonstration_2026-01-18_18:21:44.pkl', 'demonstration_2026-01-18_18:22:05.pkl', 
 'demonstration_2026-01-18_18:22:24.pkl', 'demonstration_2026-01-18_18:22:41.pkl', 'demonstration_2026-01-18_18:22:59.pkl', 'demonstration_2026-01-18_18:23:17.pkl', 'demonstration_2026-01-18_18:23:42.pkl', 
 'demonstration_2026-01-18_18:24:48.pkl', 'demonstration_2026-01-18_18:25:10.pkl', 'demonstration_2026-01-18_18:26:07.pkl', 'demonstration_2026-01-18_18:26:43.pkl', 'demonstration_2026-01-18_18:27:04.pkl', 
 'demonstration_2026-01-18_18:27:21.pkl', 'demonstration_2026-01-18_18:27:40.pkl', 'demonstration_2026-01-18_18:27:56.pkl', 'demonstration_2026-01-18_18:28:13.pkl', 'demonstration_2026-01-18_18:28:58.pkl', 
 'demonstration_2026-01-18_18:29:34.pkl', 'demonstration_2026-01-18_18:30:10.pkl', 'demonstration_2026-01-18_18:31:03.pkl', 'demonstration_2026-01-18_18:31:49.pkl', 'demonstration_2026-01-18_18:32:12.pkl', 
 'demonstration_2026-01-18_18:32:38.pkl', 'demonstration_2026-01-18_18:33:02.pkl', 'demonstration_2026-01-18_18:33:23.pkl', 'demonstration_2026-01-18_18:33:42.pkl', 'demonstration_2026-01-18_18:34:16.pkl', 
 'demonstration_2026-01-18_18:34:51.pkl', 'demonstration_2026-01-18_18:35:12.pkl', 'demonstration_2026-01-18_18:35:41.pkl', 'demonstration_2026-01-18_18:36:00.pkl', 'demonstration_2026-01-18_18:36:30.pkl', 
 'demonstration_2026-01-18_18:36:52.pkl', 'demonstration_2026-01-18_18:37:24.pkl', 'demonstration_2026-01-18_18:37:46.pkl', 'demonstration_2026-01-18_18:38:09.pkl', 'demonstration_2026-01-18_18:38:30.pkl', 
 'demonstration_2026-01-18_18:38:51.pkl', 'demonstration_2026-01-18_18:39:15.pkl', 'demonstration_2026-01-18_18:39:31.pkl', 'demonstration_2026-01-18_18:40:41.pkl', 'demonstration_2026-01-18_18:41:09.pkl', 
 'demonstration_2026-01-18_18:41:45.pkl', 'demonstration_2026-01-18_18:42:05.pkl', 'demonstration_2026-01-18_18:42:26.pkl', 'demonstration_2026-01-18_18:43:17.pkl', 'demonstration_2026-01-18_18:43:36.pkl', 
 'demonstration_2026-01-18_18:43:51.pkl', 'demonstration_2026-01-18_18:44:53.pkl', 'demonstration_2026-01-18_18:45:08.pkl', 'demonstration_2026-01-18_18:45:26.pkl', 'demonstration_2026-01-18_18:45:45.pkl', 
 'demonstration_2026-01-18_18:46:01.pkl', 'demonstration_2026-01-18_18:46:22.pkl', 'demonstration_2026-01-18_18:46:40.pkl', 'demonstration_2026-01-18_18:46:55.pkl', 'demonstration_2026-01-18_18:47:09.pkl', 
 'demonstration_2026-01-18_18:49:34.pkl', 'demonstration_2026-01-18_18:49:50.pkl', 'demonstration_2026-01-18_18:50:12.pkl', 'demonstration_2026-01-18_18:50:28.pkl', 'demonstration_2026-01-18_18:50:47.pkl', 
 'demonstration_2026-01-18_18:51:19.pkl', 'demonstration_2026-01-18_18:51:37.pkl', 'demonstration_2026-01-18_18:52:24.pkl', 'demonstration_2026-01-18_18:52:49.pkl', 'demonstration_2026-01-18_18:53:17.pkl']

for pkl in pkls:
    print(pkl.name)
    with pkl.open("rb") as f:
        demo = pickle.load(f)
    for i in range(len(demo)):
        #resize images to 84*84 and remove depth info
        img_hwc = np.transpose(demo[i]['rgb_cabine'], (1, 2, 0))
        demo[i]['rgb_cabine'] = np.transpose(cv2.resize(img_hwc, (224, 224), interpolation=cv2.INTER_AREA), (2, 0, 1))
        demo[i]['depth_cabine'] = []
        demo[i]['action'] = -50 * demo[i]['action']
        demo[i]["success"] = 0
    if pkl.name in successful_demos:
        demo[-1]["success"] = 1

    # write demo to file at dst_dataset
    with open(FLAGS.dst_dataset+"/"+pkl.name, "wb") as f:
        pickle.dump(demo, f)
    print(f"Processed {pkl.name}")
print("finished")