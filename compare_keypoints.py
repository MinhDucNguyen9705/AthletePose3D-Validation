import os
import numpy as np
import json
from tqdm import tqdm

# Define the keypoints in the order they appear in the data
KEYPOINTS = [
    'Root', 'LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle',
    'Belly', 'Neck', 'Nose', 'Head', 
    'RShoulder', 'RElbow', 'RHand', 'LShoulder', 'LElbow', 'LHand'
]

def get_all_actors_and_cameras(root_dir):
    """Get all actor and camera names from the root directory."""
    set_numbers = os.listdir(root_dir)
    actor_names = {n: dict() for n in set_numbers}
    for set_number in set_numbers:
        actor_paths = os.listdir(f"{root_dir}/{set_number}")
        for actor_path in actor_paths:
            if actor_path.split(".")[-1] != "json":
                continue
            actor_path = actor_path.split(".")[0]
            actor_name = "_".join(actor_path.split("_")[:-2])
            camera_name = "_".join(actor_path.split("_")[-2:])
            if actor_name in actor_names[set_number]:
                actor_names[set_number][actor_name].add(camera_name)
            else:
                actor_names[set_number][actor_name] = {camera_name}

    return actor_names

def get_overlap_frames(json_cam1, json_cam2):
    """Get the overlapping frame range between two camera JSON files."""
    with open(json_cam1, 'r') as f:
        data_cam1 = json.load(f)
    with open(json_cam2, 'r') as f:
        data_cam2 = json.load(f)

    cam_1_start_frame = data_cam1['source_start_frame']
    cam_1_end_frame = data_cam1['source_end_frame']
    cam_2_start_frame = data_cam2['source_start_frame']
    cam_2_end_frame = data_cam2['source_end_frame']

    overlap_start = max(cam_1_start_frame, cam_2_start_frame)
    overlap_end = min(cam_1_end_frame, cam_2_end_frame)
    # print(f"Camera 1 frames: {cam_1_start_frame} to {cam_1_end_frame}")
    # print(f"Camera 2 frames: {cam_2_start_frame} to {cam_2_end_frame}")
    # print(f"Overlap frames: {overlap_start} to {overlap_end}")

    if overlap_start <= overlap_end:
        return overlap_start, overlap_end
    else:
        return -1, -1  # No overlap

def frame_mapping(data_cam1, data_cam2, frame_number):
    """Map a frame number from one camera to the corresponding frame number in another camera."""

    cam_1_start_frame = data_cam1['source_start_frame']
    cam_1_end_frame = data_cam1['source_end_frame']
    cam_2_start_frame = data_cam2['source_start_frame']
    cam_2_end_frame = data_cam2['source_end_frame']
    # print(f"Camera 1 frames: {cam_1_start_frame} to {cam_1_end_frame}")
    # print(f"Camera 2 frames: {cam_2_start_frame} to {cam_2_end_frame}")
    # print(f"Requested frame number: {frame_number}")

    if cam_1_start_frame <= frame_number <= cam_1_end_frame and cam_2_start_frame <= frame_number <= cam_2_end_frame:
        cam_1_frame_index = frame_number - cam_1_start_frame
        cam_2_frame_index = frame_number - cam_2_start_frame
        return cam_1_frame_index, cam_2_frame_index
    else:
        raise ValueError("Frame number is out of range for one or both cameras.")

def infront_of_camera(keypoints3d):
    """
    Determine if the person is facing towards the camera based on the 3D keypoints.
    
    Args:
        keypoints3d (dict): A dictionary containing 3D coordinates of keypoints.
    Returns:
        bool: True if the person is facing towards the camera, False otherwise.
    """

    root_points = ['RShoulder', 'LShoulder', 'RHip']
    rshoulder = keypoints3d.get('RShoulder')
    lshoulder = keypoints3d.get('LShoulder')
    rhip = keypoints3d.get('RHip')
    nose = keypoints3d.get('Nose')

    v1 = np.array(lshoulder) - np.array(rshoulder)
    v2 = np.array(rhip) - np.array(rshoulder)
    front_vector = np.cross(v2, v1)
    n = np.linalg.norm(front_vector)
    front_vector = front_vector / (n + 1e-8)

    store = {}
    for point in keypoints3d.keys():
        v = np.array(keypoints3d[point]) - np.array(nose)
        v = v / (np.linalg.norm(v) + 1e-8)
        d = float(np.dot(v, front_vector))
        if d < 0:
            store[point] = -1
        else:
            store[point] = 1
    
    for point in root_points:
        if point in store:
            store[point] = 1
    return store

def distance_to_other_keypoint(keypoints3d):
    """
    Calculate the distance from each keypoint to the camera (assumed to be at the origin).
    
    Args:
        keypoints3d (dict): A dictionary containing 3D coordinates of keypoints.
    Returns:
        dict: A dictionary with keypoints as keys and their distances to the camera as values.
    """
    distances = {}
    for point1, coords1 in keypoints3d.items():
        distances[point1] = []
        for point2, coords2 in keypoints3d.items():
            if point1 != point2:
                dist = np.linalg.norm(np.array(coords1) - np.array(coords2))
                distances[point1].append((dist))
    return distances

def calculate_component_difference(keypoints3d_cam1, keypoints3d_cam2, threshold=5):
    
    point_differences = {}
    for point in keypoints3d_cam1:
        num_diff = 0
        for i in range (len(keypoints3d_cam1[point])-1):
            a = keypoints3d_cam1[point][i]
            b = keypoints3d_cam2[point][i]
            n = 2 * abs(a - b) / (abs(a) + abs(b) + 1e-8) * 100
            if n > threshold:
                num_diff += 1
        point_differences[point] = num_diff
    return point_differences

def compare_keypoints(root_dir, set_number, actor_name, video_name_cam1, video_name_cam2, frame_number):

    data_cam1 = np.load(f'{root_dir}/{set_number}/{actor_name}_{video_name_cam1}_h36m.npy', allow_pickle=True)
    data_cam2 = np.load(f'{root_dir}/{set_number}/{actor_name}_{video_name_cam2}_h36m.npy', allow_pickle=True)
    
    cam_1_json_path = f'{root_dir}/{set_number}/{actor_name}_{video_name_cam1}.json'
    cam_2_json_path = f'{root_dir}/{set_number}/{actor_name}_{video_name_cam2}.json'

    with open(cam_1_json_path, 'r') as f:
        data_cam1_json = json.load(f)
    with open(cam_2_json_path, 'r') as f:
        data_cam2_json = json.load(f)
    

    frame_cam_1, frame_cam_2 = frame_mapping(data_cam1_json, data_cam2_json, frame_number)

    keypoints3d_cam1 = {}
    keypoints3d_cam2 = {}

    for keypoint, coords_cam1, coords_cam2 in zip(KEYPOINTS, data_cam1[frame_cam_1], data_cam2[frame_cam_2]):

        keypoints3d_cam1[keypoint] = coords_cam1.tolist()
        keypoints3d_cam2[keypoint] = coords_cam2.tolist()

    result_cam1 = infront_of_camera(keypoints3d_cam1)
    distances_cam1 = distance_to_other_keypoint(keypoints3d_cam1)
    result_cam2 = infront_of_camera(keypoints3d_cam2)
    distances_cam2 = distance_to_other_keypoint(keypoints3d_cam2)

    for point in keypoints3d_cam1.keys():
        distances_cam1[point].append(result_cam1[point])
        distances_cam2[point].append(result_cam2[point])
    
    # Compare direction of keypoints between the two cameras
    count = 0
    diff_direction = []
    for point in keypoints3d_cam1.keys():
        cam1_facing = distances_cam1[point][-1]
        cam2_facing = distances_cam2[point][-1]
        if cam1_facing != cam2_facing:
            diff_direction.append(point)
    
    # print("Keypoints with different facing directions between the two cameras:")
    # for point in diff_direction:
    #     print(f"{point}: Cam1 - {distances_cam1[point][-1]}, Cam2 - {distances_cam2[point][-1]}")

    diff_array = []
    for point in keypoints3d_cam1.keys():
        if point in diff_direction:
            continue
        diff_norm = np.linalg.norm(np.array(distances_cam1[point][:-1]) - np.array(distances_cam2[point][:-1]))
        diff_array.append(diff_norm)
    
    # print("Difference in distances to camera for each keypoint:")
    # for point, diff in zip(keypoints3d_cam1.keys(), diff_array):
    #     print(f"{point}: {diff}")
    
    # Print out keypoints with difference on the Q3 quartile
    q3 = np.percentile(diff_array, 75)
    # print(f"Keypoints with distance difference above the Q3 quartile ({q3}):")
    for point, diff in zip(keypoints3d_cam1.keys(), diff_array):
        if diff > q3:
            # print(f"{point}: {diff}")
            count +=1
    count += len(diff_direction)
    # print(f"Total keypoints with significant differences: {count}")
    if count > 0:
        print(f"Actor: {actor_name}, Cam1: {video_name_cam1}, Cam2: {video_name_cam2}, Frame: {frame_number}, Significant Differences: {count}")

if __name__ == "__main__":

    root_dir = "AthletePose3D/data/test_set"
    all_data = get_all_actors_and_cameras(root_dir)

    for set_number, actors in tqdm(all_data.items(), desc="Processing sets"):
        # print(f"Set: {set_number}: {actors}")
        for actor_name, cameras in tqdm(actors.items(), desc="Processing actors"):
            for cam1 in cameras:
                for cam2 in cameras:
                    if cam1 != cam2:
                        json_cam1 = os.path.join(root_dir, f"{set_number}/{actor_name}_{cam1}.json")
                        json_cam2 = os.path.join(root_dir, f"{set_number}/{actor_name}_{cam2}.json")
                        # print(f"Comparing {actor_name} between {cam1} and {cam2}")
                        overlap_start, overlap_end = get_overlap_frames(json_cam1, json_cam2)
                        if overlap_start != -1:
                            # print(f"Comparing {actor_name} between {cam1} and {cam2} with overlapping frames from {overlap_start} to {overlap_end}")
                            for frame_number in range(overlap_start, overlap_end + 1):
                                # print(f"Comparing {actor_name} between {cam1} and {cam2} at frame {frame_number}")
                                compare_keypoints(root_dir, set_number, actor_name, cam1, cam2, frame_number)