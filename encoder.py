import torch
import torch.nn as nn
import cv2
import torchvision.models as models
import numpy as np
from vgg19 import vgg19
import os
def read_video_file(filename):
    """
    Read video from file.
    Return an array of frames (4-D array).
    """
    cap = cv2.VideoCapture(filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # original width
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # original height
    video = np.zeros((frame_count, 224, 224, 3), dtype='uint8')
    count = 0
    while cap.isOpened() and count < frame_count:
        ret, frame = cap.read() # ret = 1 if the video is captured
        if ret is True:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            video[count] = frame # The dimension of each frame is [height, width, 3]
        count += 1
        # cv2.imshow('image', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    video = np.transpose(video, (0,3,1,2))
    return video # The dimension of video is [frame_count, 3, frame_height, frame_width]

def embed(frames):
    frames = torch.from_numpy(frames)
    frames = frames.type(torch.float)
    embedding_list = []
    print("frames shape", frames.size())
    for frame in frames:
        emnedding = model(frame)
        embedding_list.append(emnedding)



def save_video_file(pathname):
    file_list = os.listdir

    pass

def main():
    directory_list = ['./data/hmdb51/smile/']
    for directory in directory_list:
        file_list = os.listdir(directory)
        for video in file_list:
            frames = read_video_file(video)



if __name__ == '__main__':
    # filename = "./data/hmdb51/smile/Smile!_smile_h_cm_np1_fr_goo_0.avi"
    filename = "./data/hmdb51/smile/A_Beautiful_Mind_2_smile_h_nm_np1_fr_goo_2.avi"
    frames = read_video_file(filename)
    print("number of frames", len(frames))

    model = vgg19()
    print(model)
    frames = torch.from_numpy(frames)
    frames = frames.type(torch.float)
    print(frames.size())
    print(frames[0].size())
    # outputs = model(torch.unsqueeze(torch.rand(3,224,224), 0))
    outputs = model(torch.unsqueeze(frames[0], 0))
    print(outputs.size())

    # print(os.listdir("./data/hmdb51/smile/"))
    print(os.listdir("./data/hmdb51/"))
