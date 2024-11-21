from torch.utils.data import Dataset
from pandas import read_parquet
import io
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from mm_utils.utils import *

class Pokemon(Dataset):
    def __init__(
        self,
        data_path = "/data3/haibo/data/train-00000-of-00001-566cc9b19d7203f8.parquet",
        img_size = 512,
    ):
        self.data = read_parquet(data_path)
        self.image_processor = image_transform(image_size=img_size, mean=0.5, std=0.5, random_flip=False)
        self.prompts = []
        self.images = []

        for idx in range(self.data.shape[0]):
            self.images.append(Image.open(io.BytesIO(self.data.loc[idx, 'image']['bytes'])))
            self.prompts.append(self.data.loc[idx, 'text'])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        caption = self.prompts[index]
        pixel_values = (self.image_processor(self.images[index])) # [3, 224, 224]
        return {
                "prompts": caption,
                "pixel_values": pixel_values,
            }

# import random
# dataset = Pokemon()
# for i in range(2):
#     entry = random.choice(dataset)
#     print(entry['pixel_values'].shape)
#     print("prompts: ", entry['prompts'])
#     print()