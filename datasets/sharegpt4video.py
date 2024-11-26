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

class ShareGPT4Video(Dataset):
    def __init__(
        self,
        anno_path = '/home/haibo/data/sharegpt4video/sharegpt4video_40k.jsonl',
        video_path = "/home/haibo/data/sharegpt4video/videos",
        num_frames = 16,
        sample='middle',
        img_size = 512,
        stride=-1,
    ):
        self.video_path = video_path
        self.num_frames = num_frames
        self.sample = sample
        self.stride = stride

        self.data = load_jsonl(anno_path)

        self.image_processor = frame_transform(image_size=img_size, mean=0.5, std=0.5)

        self.video_ids = []
        self.video_files = []
        self.text_inputs = []

        for item in self.data:
            self.video_files.append(item['video_path'])
            self.video_ids.append(item['video_id'])
            self.text_inputs.append(item['captions'][-1]['content'])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        text_input = self.text_inputs[index]
        video_file = str(self.video_files[index])

        video_path = os.path.join(self.video_path, video_file)

        pixel_values, frame_indices, fps, total_frame_num, duration = read_frames_decord(
            video_path = video_path,
            num_frames = self.num_frames,
            sample = self.sample,
            stride = self.stride,
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
            }


# dataset = ShareGPT4Video(num_frames=16, img_size=(320, 576))
# for i in range(10):
#     entry = random.choice(dataset)
#     print(entry['video_ids'])
#     print("prompts: ",        entry['prompts'])
#     print("pixel_values: ",   entry['pixel_values'].shape)
#     print()

#     # from moviepy.editor import ImageSequenceClip
#     # image = entry['pixel_values']
#     # image = (image / 2 + 0.5).clamp(0, 1).squeeze()  
#     # image = (image.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()  
#     # image = image.round().astype("uint8")  # [frame_num, 256, 256, 3]
#     # prompts = entry['prompts']
#     # video_id = entry['video_ids']
#     # frames = [frame for frame in image] 
#     # clip = ImageSequenceClip(frames, fps=4)
#     # clip.write_videofile(f"{video_id}.mp4", codec='libx264')

# print(len(dataset))