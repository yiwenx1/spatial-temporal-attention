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

def load_optical_flow(filename):
    """
    Read video from file.
    Return an array of frames (4-D array).
    """
    cap = cv2.VideoCapture(filename)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    # set the current position to the end of the video.
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    # current position of the video in milliseconds.
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) # duration of the video(ms)
    # set the current position to the start of the video.
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    # cap.set(cv2.CAP_PROP_FPS, 40) # does work, default fps remains 30
    DEFAULT_FPS = 30
    NUM_FRAME = 20
    num_total_frame = duration / 1000 * 30
    sample_bin = num_total_frame // NUM_FRAME
    optical_flow = np.zeros((NUM_FRAME, 224, 224), dtype='uint8')
    count = 1

    hsv[...,1] = 255
    while True:
        ret, frame2 = cap.read() # ret = 1 if the video is captured
        if ret == False:
            break
        if sample_bin == 0:
            print(count)
            next = cv2.cvtColor(frame2, cv2.COLOR_BAYER_BG2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,1], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
            prvs = next

            index = count - 1
            optical_flow[index] = bgr
        elif count % sample_bin == 0:
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
            prvs = next

            index = int(count // sample_bin) - 1
            if index < NUM_FRAME:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                optical_flow[index] = bgr # The dimension of each frame is [height, width, 3]
        count += 1
    optical_flow = np.transpose(optical_flow, (0,3,1,2))
    return optical_flow # The dimension of video is [frame_count, frame_height, frame_width]

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
    print(len(splitMap))
    return splitMap, values.count(0), values.count(1), values.count(2)

def testH5py():
    test_data = h5py.File("test.hdf5", "w")
    test_data.create_dataset("dataset", (100))

def main():
    splitMap, val_num, train_num, test_num = makeSplitMap()
    # train_num: 5236
    # test_num: 1550
    train_num = train_num + val_num

    train_file = h5py.File("train_data.hdf5", "w")
    train_data = train_file.create_dataset("train_data", (train_num, 20, 3, 224, 224), dtype = 'uint8')
    train_label_file = h5py.File("train_label.hdf5", "w")
    train_label = train_label_file.create_dataset("train_label", (train_num, ), dtype = 'i')

    val_file = h5py.File("val_data.hdf5", "w")
    val_data = val_file.create_dataset("val_data", (test_num, 20, 3, 224, 224), dtype = "uint8")
    val_label_file = h5py.File("val_label.hdf5", "w")
    val_label = val_label_file.create_dataset("val_label", (test_num, ), dtype = 'i')

    train_idx, val_idx = 0, 0
    maindir = "./data/hmdb51"
    for action in os.listdir(maindir):
        if action == '.DS_Store':
            continue
        videoDir = maindir + "/" + action
        for video in os.listdir(videoDir):
            if video == '.DS_Store':
                continue

            data_type = splitMap[video]
            print(video)
            feature = read_video_file(videoDir + "/" + video)
            label = LABELS.index(action)
            if data_type == 0 or data_type == 1: # train + val
                train_data[train_idx] = feature
                train_label[train_idx] = label
                train_idx += 1
            elif data_type == 2: # test
                val_data[val_idx] = feature
                val_label[val_idx] = label
                val_idx += 1
    train_file.close()
    train_label_file.close()
    val_file.close()
    val_label_file.close()
    print("done")

if __name__ == '__main__':
    # main()
    filename = "./data/hmdb51/swing_baseball/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_14.avi"
    optical_flow = load_optical_flow(filename)
