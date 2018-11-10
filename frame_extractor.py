import cv2
import numpy as np
import os
import subprocess
import h5py


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
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) # duration of the video(ms) 
    # set the current position to the start of the video.      
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0) 
    # cap.set(cv2.CAP_PROP_FPS, 20) # does work, default fps remains 30   
    DEFAULT_FPS = 30
    NUM_FRAME = 20
    num_total_frame = duration / 1000 * 30
    sample_bin = num_total_frame // NUM_FRAME
    video = np.zeros((NUM_FRAME, 224, 224, 3), dtype='uint8')
    count = 1 
    # while cap.isOpened() and count < frame_count:
    while True:   
        ret, frame = cap.read() # ret = 1 if the video is captured    
        if ret == False:
            break
        if count % sample_bin == 0:          
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)  
            index = int(count // sample_bin) - 1
            if index < NUM_FRAME:          
                video[index] = frame # The dimension of each frame is [height, width, 3]
            # cv2.imshow('image', frame)            
            # cv2.waitKey(0)            
            # cv2.destroyAllWindows()        
        count += 1   
    video = np.transpose(video, (0,3,1,2))    
    return video # The dimension of video is [frame_count, 3, frame_height, frame_width]

# the split file from HDBM51 website: http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
# 1: training
# 2: testing
# 0: validation
def makeSplitMap():
    splitMap = dict()
    maindir = "./testTrainMulti_7030_splits"
    for split_file in os.listdir(maindir):
        if split_file == '.DS_Store':
            continue
        f = open(maindir + "/" + split_file, "r").read()
        for line in f.splitlines():
            line = line.split(" ")
            splitMap[line[0]] = int(line[1])
    values = list(splitMap.values())
    return splitMap, values.count(0), values.count(1), values.count(2)

def testH5py():
    test_data = h5py.File("test.hdf5", "w")
    test_data.create_dataset("dataset", (100))

def main():
    splitMap, val_num, train_num, test_num = makeSplitMap()

    train_data = h5py.File("train_data.hdf5", "w").create_dataset("train_data", (train_num, 20, 3, 224, 224))
    train_label = h5py.File("train_label.hdf5", "w").create_dataset("train_label", (train_num, ))
    
    val_data = h5py.File("val_data.hdf5", "w").create_dataset("val_data", (val_num, 20, 3, 224, 224))
    val_label = h5py.File("val_label.hdf5", "w").create_dataset("val_label", (val_num, ))
    
    test_data = h5py.File("test_data.hdf5", "w").create_dataset("test_data", (test_num, 20, 3, 224, 224))
    test_label = h5py.File("test_label.hdf5", "w").create_dataset("test_label", (test_num, ))


    train_idx, val_idx, test_idx = 0, 0, 0
    maindir = "./data" 
    for action in os.listdir(maindir):
        # print("here 1")
        if action == '.DS_Store':
            continue
        print(action)
        videoDir = maindir + "/" + action
        for video in os.listdir(videoDir):
            if video == '.DS_Store':
                continue
            
            data_type = splitMap[video]
            feature = read_video_file(videoDir + "/" + video)
            label = LABELS.index(action)
            if data_type == 0: # val
                val_data[val_idx] = feature
                val_label[val_idx] = label
                val_idx += 1
            elif data_type == 1: # train
                train_data[train_idx] = feature
                train_label[train_idx] = label
                train_idx += 1
            elif data_type == 2: # test
                test_data[test_idx] = feature
                test_label[test_idx] = label
                test_idx += 1
    print("done")

if __name__ == '__main__':
    main()
