import os
import subprocess
from pathlib import Path
from glob import glob
import cv2
from pydub import AudioSegment
import imageio
import numpy as np
from alignment import dlib_alignment
import re

#to create jpg frames of original video 
def original_video_to_frames(video_path, frames_dir):
    
    if os.path.exists(frames_dir):
        frames_paths = _find_filenames(Path(frames_dir), '*.jpg')
        if len(frames_paths) > 1:
            print("(frames exists) Skipping .....")
            return
    
    frames_save_string = frames_dir+ "/%07d.jpg"
    
    subprocess.call(['ffmpeg',  '-noautorotate', '-i', video_path, '-f', 'image2', '-q:v', '1', frames_save_string])
    

#search files and their paths
def _find_filenames(file_dir_path, file_pattern): return list(
	file_dir_path.glob(file_pattern))

#get video fps
def get_video_fps(video_path):
    video = imageio.get_reader(video_path)
    fps = video.get_meta_data()['fps']
    return fps

#load cropped_frames
def load_cropped_frames(frames_dir, final_img_size = 96):
    frames_paths = _find_filenames(Path(frames_dir), '*.jpg')
    frames_paths = [str(filename) for filename in frames_paths]
    full_frames = []
    
    for file_path in frames_paths:
        frame = cv2.imread(file_path, 1)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (final_img_size, final_img_size), interpolation=cv2.INTER_AREA)
        full_frames.append({'frame':frame, 'path':file_path})
        frames_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        
    return full_frames


#load cropped_frames
#load cropped_frames
def load_align_frames(frames_dir, original_frames_dir, landmarks, save_dir, final_img_size = 96):
    frames_paths = _find_filenames(Path(frames_dir), '*.jpg')
    frames_paths = [str(filename) for filename in frames_paths]
    full_frames = []
    
    for file_path in frames_paths:
        
        org_path = os.path.join(original_frames_dir, os.path.basename(file_path))
        
        frame = cv2.imread(org_path, 1)
        
        landmark = landmarks[os.path.basename(file_path)]
        landmark = np.asarray(landmark, dtype=np.float32)
        img, M = dlib_alignment(frame, landmark, padding=0.18, size=512, moving=-0.1)
        save_str = os.path.join(save_dir,os.path.basename(file_path))
        print(save_str)
        cv2.imwrite(save_str, img)

    

def load_frame(frame_path, resize_factor=1, rotate=False, crop = [0, -1, 0, -1]):
    #print("reading original frames .....")
    
    frame = cv2.imread(frame_path, 1)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if resize_factor > 1:
        frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
            
    if rotate:
        frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            
    y1, y2, x1, x2 = crop
    if x2 == -1: x2 = frame.shape[1]
    if y2 == -1: y2 = frame.shape[0]
        
    frame = frame[y1:y2, x1:x2]
        
    return frame

#load frames from dir in Memory 
def load_frames_paths(frames_dir, resize_factor=1, rotate=False, crop = [0, -1, 0, -1]):
    #print("reading original frames .....")
    frames_paths = _find_filenames(Path(frames_dir), '*.jpg')
    frames_paths = [str(filename) for filename in frames_paths]
    frames_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    frames_paths=frames_paths 
    return frames_paths


def load_frames_for_stiching(full_frames):
    results = {}
    
    for obj in full_frames:
        filename = os.path.basename(obj['path'])
        results[filename] = obj['frame']
        
    return results

#load/extract audio from local
def load_or_extract_audio_from_local(video_path, audio_path):
    if os.path.exists(audio_path):
        print("(audio exists) Skipping .....")
    else:
        print("extracting audio from original_video")
        subprocess.call(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path])
    
    audio = AudioSegment.from_mp3(audio_path)
    
    return audio


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'