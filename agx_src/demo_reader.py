# import cv2
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

def get_stratified_demos_list(all_demos, demos_ratio):
    # return list with fixed ratio of successful and unsuccessful demos
    # hardcoding success and failure lists
    # Can later use success field or provide lists via config
    unsuccessful_demos = [
    "demonstration_2026-01-18_18:11:01.pkl",
    "demonstration_2026-01-18_18:11:08.pkl",
    "demonstration_2026-01-18_18:23:53.pkl",
    "demonstration_2026-01-18_18:40:19.pkl",
    "demonstration_2026-01-18_18:11:03.pkl",
    "demonstration_2026-01-18_18:21:00.pkl",
    "demonstration_2026-01-18_18:39:43.pkl",
    "demonstration_2026-01-18_18:47:35.pkl",
    "demonstration_2026-01-18_18:40:01.pkl",
    "demonstration_2026-01-18_18:48:54.pkl",
    "demonstration_2026-01-18_18:11:06.pkl",
    "demonstration_2026-01-18_18:44:10.pkl",
    "demonstration_2026-01-18_18:49:20.pkl",
    "demonstration_2026-01-18_18:30:36.pkl",
    "demonstration_2026-01-18_18:31:17.pkl",
    "demonstration_2026-01-18_18:23:50.pkl",
    "demonstration_2026-01-18_18:31:35.pkl",
    "demonstration_2026-01-18_18:52:07.pkl",
    "demonstration_2026-01-18_18:10:58.pkl",
    "demonstration_2026-01-18_18:44:28.pkl",
    "demonstration_2026-01-18_18:23:56.pkl"
    ]


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

    # initialize sampled demos
    sampled_demos = ['' for i in range(round(len(all_demos)*demos_ratio/100))]

    fail_ctr = 0
    success_ctr = 0
    target_ratio = len(unsuccessful_demos) / len(all_demos)
    while fail_ctr / len(sampled_demos) < target_ratio:
        #fill failed demos
        sampled_demos[fail_ctr] = unsuccessful_demos[fail_ctr]
        fail_ctr += 1
    # now fill the rest with successful demos
    while success_ctr < len(sampled_demos) - fail_ctr:
        sampled_demos[fail_ctr+success_ctr] = successful_demos[success_ctr]
        success_ctr += 1

    return sampled_demos


