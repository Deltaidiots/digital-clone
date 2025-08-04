import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path):
    
    
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    os.makedirs(full_body_dir, exist_ok=True)

    total_frames = get_video_frame_count(path)
    existing_frames = len([name for name in os.listdir(full_body_dir) if os.path.isfile(os.path.join(full_body_dir, name))])

    if total_frames > 0 and existing_frames >= total_frames:
        print(f"Images in '{full_body_dir}' seem to be completely extracted already ({existing_frames}/{total_frames} frames). Skipping image extraction.")
        return
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        # High quality conversion to 25fps using ffmpeg
        cmd = f'ffmpeg -i {path} -vf "fps=25" -c:v libx264 -c:a aac {path.replace(".mp4", "_25fps.mp4")}'
        os.system(cmd)
        path = path.replace(".mp4", "_25fps.mp4")
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError("Your video fps should be 25!!!")
        
    print("extracting images...")
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(full_body_dir+"/"+str(counter)+'.jpg', frame)
        counter += 1
        
def get_audio_feature(wav_path):
    
    print("extracting audio feature...")
    base_dir = os.path.dirname(wav_path)
    audio_feature_path = os.path.join(base_dir, 'aud_ave.npy')
    if os.path.exists(audio_feature_path):
        print(f"Audio feature file '{audio_feature_path}' already exists. Skipping audio feature extraction.")
        return

    os.system("python ./data_utils/ave/test_w2l_audio.py --wav_path "+wav_path)
    
def get_landmark(path, landmarks_dir):
    print("detecting landmarks...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")

    if not os.path.exists(full_img_dir) or len(os.listdir(full_img_dir)) == 0:
        print(f"Image directory '{full_img_dir}' is empty. Cannot process landmarks.")
        return

    num_images = len([name for name in os.listdir(full_img_dir) if name.endswith('.jpg')])
    
    os.makedirs(landmarks_dir, exist_ok=True)
    num_landmarks = len([name for name in os.listdir(landmarks_dir) if name.endswith('.lms')])

    if num_images > 0 and num_landmarks >= num_images:
        print(f"Landmarks in '{landmarks_dir}' seem to be completely generated already ({num_landmarks}/{num_images} files). Skipping landmark detection.")
        return
    
    from get_landmark import Landmark
    landmark = Landmark()
    
    for img_name in tqdm(os.listdir(full_img_dir)):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        pre_landmark, x1, y1 = landmark.detect(img_path)
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0]+x1, p[1]+y1
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path)

