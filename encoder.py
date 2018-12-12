import torch
import torch.nn as nn
import cv2
import torchvision.models as models
import numpy as np
import os

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
    # cap.set(cv2.CAP_PROP_FPS, 40) # does work, default fps remains 30
    DEFAULT_FPS = 30
    NUM_FRAME = 20
    num_total_frame = duration / 1000 * 30
    sample_bin = num_total_frame // NUM_FRAME
    video = np.zeros((NUM_FRAME, 224, 224, 3), dtype='uint8')
    count = 1
    if sample_bin == 0:
        print(filename)
    # while cap.isOpened() and count < frame_count:
    while True:
        ret, frame = cap.read() # ret = 1 if the video is captured
        if ret == False:
            break
        if sample_bin == 0:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            index = count - 1
            video[index] = frame
        elif count % sample_bin == 0:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            index = int(count // sample_bin) - 1
            if index < NUM_FRAME:
                video[index] = frame # The dimension of each frame is [height, width, 3]
                cv2.imshow('image', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite(str(index)+'.png', frame)
        count += 1
    video = np.transpose(video, (0,3,1,2))
    return video # The dimension of video is [frame_count, 3, frame_height, frame_width]

def embed(frames, label):
    vgg19 = models.vgg19(pretrained=True)
    vgg19.eval()
    extractor = nn.Sequential(*list(vgg19.features.children()))[:-1]

    frames = torch.from_numpy(frames)
    frames = frames.type(torch.float)
    embedding_list = []
    print("frames shape", frames.size())
    for frame in frames:
        emnedding = extractor(frame.unsqueeze(dim=0))
        embedding_list.append(emnedding)
    embedding_list.append(LABELS.index(label))
    return embedding_list


def save_video_file(pathname):
    pass

def main():
    directory_list = ['./data/hmdb51/ride_bike/']
    # directory_list = ['./data/hmdb51/ride_bike/',
    #                   '.data/hmdb51/ride_horse/',
    #                   './data/hmdb51/run/',
    #                   './data/hmdb51/shake_hands/',
    #                   './data/hmdb51/shoot_ball/',
    #                   './data/hmdb51/shoot_bow/',
    #                   './data/hmdb51/shoot_gun/',
    #                   './data/hmdb51/sit/',
    #                   './data/hmdb51/situp/',
    #                   './data/hmdb51/smile/',
    #                   './data/hmdb51/smoke/',
    #                   './data/hmdb51/somersault/',
    #                   './data/hmdb51/stand/',
    #                   './data/hmdb51/swing_baseball/',
    #                   './data/hmdb51/sword/',
    #                   './data/hmdb51/sword_exercise/',
    #                   './data/hmdb51/talk/',
    #                   './data/hmdb51/throw/',
    #                   './data/hmdb51/turn/',
    #                   './data/hmdb51/walk/',
    #                   './data/hmdb51/wave/']
    for directory in directory_list:
        file_list = os.listdir(directory)
        file_list = file_list[:1]
        print(file_list)
        for video_file in file_list:
            video_file = directory + video_file
            frames = read_video_file(video_file)
            embedding_list = embed(frames, directory[14:-1])
            print(embedding_list[0].shape, len(embedding_list), embedding_list[-1])


if __name__ == '__main__':
    filename = "./data/hmdb51/swing_baseball/practicingmybaseballswing2009_swing_baseball_f_cm_np1_fr_med_14.avi"
    frames = read_video_file(filename)
    # print("number of frames", frames.shape)

    # model = vgg19()
    # frames = torch.from_numpy(frames)
    # frames = frames.type(torch.float)
    # print(frames.size())
    # print(frames[0].size())
    # # outputs = model(torch.unsqueeze(torch.rand(3,224,224), 0))
    # outputs = model(torch.unsqueeze(frames[0], 0))
    # print(outputs.size())

    # print(os.listdir("./data/hmdb51/"))
    # main()
