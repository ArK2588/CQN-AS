import cv2
import numpy as np

def resize_chw_image(img, out_size):
    # converting img to hwc
    img_hwc = np.transpose(img, (1, 2, 0))
    resized_img = np.transpose(cv2.resize(img_hwc, out_size, interpolation=cv2.INTER_AREA), (2, 0, 1))
    return resized_img

def standardize_to_agxenv(step):
        # if already in agx env format ignore
        if "policy" in step and "camera" in step:
            return step

        out = dict(step)

        # renames
        if "policy" not in out and "state" in out:
            out["policy"] = out["state"]
        if "bucket" not in out and "bucket_pos" in out:
            out["bucket"] = out["bucket_pos"]
        if "cabin_position" not in out and "cabin_pos" in out:
            out["cabin_position"] = out["cabin_pos"]
        if "stone" not in out and "stone_pos" in out:
            out["stone"] = out["stone_pos"]

        # nesting and renames
        cam = dict(out.get("camera", {}))
        if "rgb" not in cam and "rgb_cabine" in out:
            cam["rgb"] = out["rgb_cabine"]
        if "depth" not in cam and "depth_cabine" in out:
            cam["depth"] = out["depth_cabine"]
        out["camera"] = cam

        return out

