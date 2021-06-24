
import os
import json
#from .lipsync import wav2lip_frames_inference
from conversions import get_smoothened_boxes
from conversions import get_bounding_box_from_landmarks
from utils_dubsync import load_align_frames


def transform_trajectories_frames_to_lipsync(save_dir):
    
    
    trajectories_save_string = save_dir+'/trajectories.json'
    original_frames_dir = save_dir+'/original_video_frames'
    
    with open(trajectories_save_string) as trajectories_file:
            trajectories = json.load(trajectories_file)
            
    for trajectory in trajectories:
        print("Wav2lip: {}".format(trajectory))
        trajectory_dir = trajectory['path']
        
        frames_dir = trajectory_dir+"/frames"
        
        lipsync_align_dir = trajectory_dir+"/wav2lip_align"
        
        if not os.path.exists(lipsync_align_dir):
            os.makedirs(lipsync_align_dir, 0o777)
        
        align_frames_dir = trajectory_dir+"/align/"
        
        audio_file = trajectory_dir+"/audio.wav"
        
        if not os.path.exists(trajectory_dir+"/align"):
            os.makedirs(trajectory_dir+"/align", 0o777)
        
        save_dir = trajectory_dir+'/wav2lip'

        recover_dir = trajectory_dir+'/wav2lip_recover'

        if not os.path.exists(recover_dir):
            os.makedirs(recover_dir, 0o777)
        
        save_dir_align = trajectory_dir+'/wav2lip_align_final'
        
        crop_file = trajectory_dir+"/wav2lip.mp4"
        
        outfile = trajectory_dir+"/synced.mp4"
        
        save_dir_hq_align = trajectory_dir+'/hq_align'
        
        if not os.path.exists(save_dir_hq_align):
            os.makedirs(save_dir_hq_align, 0o777)
        
        trajectory_landmarks_file = trajectory_dir+'/landmarks.json'
        
        # load landmarks for frames from landmarks.json
        with open(trajectory_landmarks_file) as landmarks_file:
            landmarks = json.load(landmarks_file)
            
        bbox_only = [get_bounding_box_from_landmarks(landmark) for filename, landmark in landmarks.items()]
        smooth_bbox = get_smoothened_boxes(bbox_only, T=5)
        org_landmarks_frame_names = list(landmarks.keys())
        bbox_smooth = {org_landmarks_frame_names[i]: smooth_bbox[i]
                     for i in range(len(org_landmarks_frame_names))}
            
        landmarks_only = [landmark for filename, landmark in landmarks.items()]
        smooth_landmarks = get_smoothened_boxes(landmarks_only, T=5)
        landmark_frame_names = list(landmarks.keys())
        landmarks_smooth = {landmark_frame_names[i]: smooth_landmarks[i]
                     for i in range(len(landmark_frame_names))}
        
        load_align_frames(frames_dir, original_frames_dir, landmarks_smooth, save_dir_hq_align)
        
       
if __name__ == "__main__":
    video_dir = "/home/ubuntu/iSeeBetter/dataset/-41gOdq2lN4_023.923900-027.594233"
    transform_trajectories_frames_to_lipsync(video_dir)