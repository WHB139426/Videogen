from torch.utils.data import Dataset
import random
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import pickle
import sys
import os
import requests
from collections import Counter
from io import BytesIO
import json
import cv2
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *
from mm_utils.video_utils import read_frames_decord, read_frames_av

class Webvid_motion(Dataset):
    def __init__(
        self,
        anno_path = "/home/haibo/data/webvid10m_motion/webvid10m_motion.csv",
        video_path = "/home/haibo/data/webvid10m_motion/videos",
        num_frames = 16,
        sample='middle',
        img_size = 512,
        stride=-1,
    ):
        self.video_path = video_path
        self.num_frames = num_frames
        self.stride = stride
        self.sample = sample

        self.data = load_csv(anno_path)
        self.image_processor = frame_transform(image_size=img_size, mean=0.5, std=0.5)

        self.video_ids = []
        self.video_files = []
        self.prompts = []

        for item in self.data:
            video_id = str(item['videoid'])
            page_dir = str(item['page_dir'])
            self.video_files.append(f'{page_dir}/{video_id}.mp4')
            self.video_ids.append(f'{page_dir}-{video_id}')
            self.prompts.append(item['name'])

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        prompt = self.prompts[index]
        video_file = str(self.video_files[index])
        video_path = os.path.join(self.video_path, video_file)
        
        try:
            pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
                video_path = video_path,
                num_frames = self.num_frames,
                sample = self.sample,
                stride = self.stride,
            )
        except Exception:
            print(f"read_frames_decord ERROR: {video_id}, {video_file}, {prompt}")
            pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
                video_path = '/home/haibo/data/msrvttqa/videos/video0.mp4',
                num_frames = self.num_frames,
                sample = self.sample,
                stride = self.stride,
            )
            prompt = "A man silently narrates his experience driving an audi"

        video_pixel_values = []
        for i in range(pixel_values.shape[0]): 
            video_pixel_values.append(self.image_processor(pixel_values[i]))
        video_pixel_values = torch.tensor(np.array(video_pixel_values)) # [num_frames, 3, 512, 512]

        if video_pixel_values.shape[0] == 1:
            video_pixel_values = video_pixel_values[0] 

        return {
                "video_ids": video_id,
                "prompts": prompt,
                "pixel_values": video_pixel_values,
            }

# dataset = Webvid_motion(num_frames=16, img_size=256, stride=8)
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['video_ids'])
#     print("prompts: ",        entry['prompts'])
#     print("pixel_values: ",   entry['pixel_values'].shape)
#     print()
# print(len(dataset))