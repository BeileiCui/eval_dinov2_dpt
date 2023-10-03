import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import imageio
import json
import cv2

class SCAREDDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        target_W,
        target_H,
        cap_depth=150,
    ):
        self.root = root

        self.split = split
        self.cap_depth = cap_depth
        
        split_path = os.path.join(self.root, "splits", "{}_files.txt".format(self.split))
        
        self.scans = []
        with open(split_path, "r") as f:
            lines = f.readlines()
            lines.sort()
            for line in lines:
                d_k, frame_id_side = line.split("\t", 1)
                frame_id, side = frame_id_side.split("\t", 1)
                
                sequence = d_k[7]
                keyframe = d_k[-1]
                side = side[0]
                
                data_splt = "train" if int(sequence) < 8 else "test"
                self.scans.append({
                        "data_splt": data_splt,
                        "sequence": sequence,
                        "keyframe": keyframe,
                        "frame_id": "{:06d}".format(int(frame_id)),
                        "side": side
                    })
        self.img_W = target_W
        self.img_H = target_H

        print("type:",self.split, "lengths", len(self.scans))


    def __getitem__(self, index):
        scan = self.scans[index]        
        data_splt = scan['data_splt']
        sequence = scan['sequence']
        keyframe = scan['keyframe']
        frame_id = scan['frame_id']
        side = scan['side']
        

        img_path = os.path.join(self.root, data_splt, "dataset_" + sequence , "keyframe_" + keyframe, "data", "left", "frame_data{}.png".format(frame_id))
        depth_path = os.path.join(self.root, data_splt, "dataset_" + sequence , "keyframe_" + keyframe, "data", "left_depth", "scene_points{}.tiff".format(frame_id))

        img = self.read_rgb(img_path, self.img_W, self.img_H)
        depth = self._read_depth(depth_path, self.cap_depth)
            
        data = {
            "sequence": sequence,
            "keyframe": keyframe,
            "frame_id": frame_id,

            "image": img,
            "depth": depth
        }
        return data

    def __len__(self):
        return len(self.scans)


    def read_rgb(self, path, img_W, img_H, aug=False):
        
        # img = cv2.imread(path)
        # img = cv2.resize(img, (img_W, img_H), interpolation=cv2.INTER_CUBIC)
        
        # if img.ndim == 2:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        img = Image.open(path).convert("RGB")
        # # print(img.size)  # (W, H)
        img = img.resize((img_W, img_H), Image.Resampling.BILINEAR) 
        # # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0      

        return img


    @staticmethod
    def _read_depth(depth_filename, cap_depth=150):

        depth = cv2.imread(depth_filename,2)
        depth = np.asarray(depth)
        depth[depth > cap_depth] = cap_depth

        return depth
        

if __name__ == "__main__":
    root = '/mnt/data-hdd2/Beilei/Dataset/SCARED'

    ds = SCAREDDataset(
        "test",
        root,
        target_W = 384,
        target_H = 384
    )
    for i in tqdm(range(len(ds))):
        ds[i]
        if i % 100 == 0:
            print(i)

