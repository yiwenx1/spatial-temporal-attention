import cv2
import numpy as np
import os
import subprocess


LABELS = ['brush_hair',
 'cartwheel',
 'catch',
 'chew',
 'clap',
 'climb',
 'climb_stairs',
 'dive',
 'draw_sword',
 'dribble',
 'drink',
 'eat',
 'fall_floor',
 'fencing',
 'flic_flac',
 'golf',
 'handstand',
 'hit',
 'hug',
 'jump',
 'kick',
 'kick_ball',
 'kiss',
 'laugh',
 'pick',
 'pour',
 'pullup',
 'punch',
 'push',
 'pushup',
 'ride_bike',
 'ride_horse',
 'run',
 'shake_hands',
 'shoot_ball',
 'shoot_bow',
 'shoot_gun',
 'sit',
 'situp',
 'smile',
 'smoke',
 'somersault',
 'stand',
 'swing_baseball',
 'sword',
 'sword_exercise',
 'talk',
 'throw',
 'turn',
 'walk',
 'wave']

def read_video_file(filename):    
    """    
    Read video from file.    
    Return an array of frames (4-D array).    
    """    
    cap = cv2.VideoCapture(filename)    
    # set the current position to the end of the video.    
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)    
    # current position of the video in milliseconds.    
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) # duration of the video    
    # set the current position to the start of the video.    
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    cap.set(cv2.CAP_PROP_FPS, 20) # set frame per second.    
    frame_count = int(duration / 1000 * 20)    
    video = np.zeros((frame_count, 224, 224, 3), dtype='uint8')    
    count = 0    
    while cap.isOpened() and count < frame_count:        
        ret, frame = cap.read() # ret = 1 if the video is captured        
        if ret is True:            
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)            
            video[count] = frame # The dimension of each frame is [height, width, 3]            
        # cv2.imshow('image', frame)            
        # cv2.waitKey(0)            
        # cv2.destroyAllWindows()        
        count += 1    
    video = np.transpose(video, (0,3,1,2))    
    return video # The dimension of video is [frame_count, 3, frame_height, frame_width]

def main():
    data, targets = list(), list()
    for rar_file in os.listdir("./data"):
        if rar_file.endswith("rar"):
            subprocess.call(['unrar', 'x', "./data/" + rar_file])
            label = rar_file[:-4]
            for video_file in os.listdir(label):
                if video_file.endswith("avi"):
                    data.append(read_video_file(label + "/" + video_file))
                    targets.append(LABELS.index(label))
    data = np.array(data)
    targets = np.array(targets)
    np.save("data.npy", data)
    np.save("targets.nap", targets)
    return data, targets
