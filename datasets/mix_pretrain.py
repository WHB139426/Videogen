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

class MixPretrain(Dataset):
    def __init__(
        self,
        anno_path = "/data3/haibo/data/mix_pretrain/mix_pretrain.json",
        video_path = "/data3/haibo/data",
        num_frames = 16,
        sample='middle',
        img_size = 512,
    ):
        self.video_path = video_path
        self.num_frames = num_frames
        self.sample = sample

        self.data = load_json(anno_path)

        self.image_processor = frame_transform(image_size=img_size, mean=0.5, std=0.5)

        self.video_ids = []
        self.video_files = []
        self.text_inputs = []
        self.dataset_names = []

        for item in self.data:
            self.video_files.append(item['video_file'])
            self.video_ids.append(item['video_id'])
            self.text_inputs.append(item['conversation'][1]['value'])
            self.dataset_names.append(item['dataset_name'])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        text_input = self.text_inputs[index]
        video_file = str(self.video_files[index])
        dataset_name = self.dataset_names[index]

        video_path = os.path.join(self.video_path, video_file)

        pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
            video_path = video_path,
            num_frames = self.num_frames,
            sample = self.sample,
        )
 
        video_pixel_values = []
        for i in range(pixel_values.shape[0]): 
            video_pixel_values.append(self.image_processor(pixel_values[i]))
        video_pixel_values = torch.tensor(np.array(video_pixel_values)) # [num_frames, 3, 512, 512]

        if video_pixel_values.shape[0] == 1:
            video_pixel_values = video_pixel_values[0] 

        return {
                "video_ids": video_id,
                "prompts": text_input,
                "pixel_values": video_pixel_values,
                "dataset_names": dataset_name,
            }

# dataset = MixPretrain(num_frames=1, img_size=256)
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['video_ids'], entry['dataset_names'])
#     print("prompts: ",        entry['prompts'])
#     print("pixel_values: ",   entry['pixel_values'].shape)
#     print()
# print(len(dataset))