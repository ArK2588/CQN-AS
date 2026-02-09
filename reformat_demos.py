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

for pkl in pkls:
    print(pkl.name)
    with pkl.open("rb") as f:
        demo = pickle.load(f)
    for i in range(len(demo)):
        # resize images to 84*84 and remove depth info
        img_hwc = np.transpose(demo[i]['rgb_cabine'], (1, 2, 0))
        demo[i]['rgb_cabine'] = np.transpose(cv2.resize(img_hwc, (84, 84), interpolation=cv2.INTER_AREA), (2, 0, 1))
        demo[i]['depth_cabine'] = []
    # write demo to file at dst_dataset
    with open(FLAGS.dst_dataset+"/"+pkl.name, "wb") as f:
        pickle.dump(demo, f)
    print(f"Processed {pkl.name}")
print("finished")