# AthletePose3D Dataset

You can download the AthletePose3D from the following link: [AthletePose3D Dataset ](https://drive.google.com/drive/folders/10YnMJAluiscnLkrdiluIeehNetdry5Ft)

## Expected data layout

The script expects this directory structure:

- `/AthletePose3D/`
  - `/data/`                      (video and motion data)
    - `/train_set/`
      - `/S1/`                      (subject)
        - `Axel_1_cam_1.mp4`      (video file)
        - `Axel_1_cam_1.json`     (video and motion information)
        - `Axel_1_cam_1.npy`      (motion data)
        - `Axel_1_cam_1_coco.npy` (COCO keypoints)
        - `Axel_1_cam_1_h36m.npy` (H3.6M keypoints)
      - `/S2/`
      - ...
    - `/valid_set/`
    - `/test_set/`
  - `/pose_2d/`                   (2D pose estimation ready data)
    - `/annotations`/               (Annotations in COCO Format)
      - `train_set.json`
      - ...
    - `/det_result/`                (Detected with YOLOv8)
      - `ap2d_train_det.json`
      - ...
    - `/train_set/`                 (Image files)          
    - `/valid_set/`
    - `/test_set/`
  - `/pose_3d/`                   (3D pose estimation ready data)   
    - `/frame_81/`                
    - `train.pkl`
    - `valid.pkl`
  - `cam_param.json`              (camera parameters)

Where each JSON file includes:

- `source_start_frame`
- `source_end_frame`

And each NPY file contains frame-wise 3D keypoints with shape compatible with:

`[num_frames, 17, 3]`