from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np
import torch

class trackNetDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640):
        self.path_dataset = os.path.abspath('./datasets/trackNet')  # Get absolute path
        assert mode in ['train', 'val'], 'incorrect mode'
        self.data = pd.read_csv(os.path.join(self.path_dataset, f'labels_{mode}.csv'))
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))         
        self.height = input_height
        self.width = input_width
        
        # Clean up the paths - remove any NaN values and ensure string type
        path_columns = ['path1', 'path2', 'path3', 'gt_path']
        for col in path_columns:
            self.data[col] = self.data[col].fillna('').astype(str)
            # Convert forward slashes to OS-specific separator
            self.data[col] = self.data[col].apply(lambda x: os.path.normpath(x) if x else '')
        
        # Convert coordinate columns to float, filling NaN with -1
        self.data['X'] = self.data['X'].fillna(-1).astype(float)
        self.data['Y'] = self.data['Y'].fillna(-1).astype(float)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Get paths
        path1 = self.data.iloc[idx]['path1']
        path2 = self.data.iloc[idx]['path2']
        path3 = self.data.iloc[idx]['path3']
        gt_path = self.data.iloc[idx]['gt_path']
        
        # Skip if any path is empty
        if not all([path1, path2, path3, gt_path]):
            # Return zero tensors if paths are missing
            inputs = torch.zeros((9, self.height, self.width), dtype=torch.float32)
            outputs = torch.zeros(self.width * self.height, dtype=torch.float32)
            return inputs, outputs, -1, -1, 0
        
        # Get coordinates and visibility
        x = float(self.data.iloc[idx]['X'])
        y = float(self.data.iloc[idx]['Y'])
        vis = int(self.data.iloc[idx]['Visibility Class'])
        
        # Construct full paths using os.path.join
        path1 = os.path.join(self.path_dataset, path1)
        path2 = os.path.join(self.path_dataset, path2)
        path3 = os.path.join(self.path_dataset, path3)
        gt_path = os.path.join(self.path_dataset, gt_path)
        
        try:
            # Read and process images
            inputs = self.get_input(path1, path2, path3)
            outputs = self.get_output(gt_path)
            
            # Convert to torch tensors with correct types
            inputs = torch.from_numpy(inputs).float()
            outputs = torch.from_numpy(outputs).float()
            
            return inputs, outputs, x, y, vis
        except Exception as e:
            print(f"Error loading images for index {idx}: {str(e)}")
            # Return zero tensors on error
            inputs = torch.zeros((9, self.height, self.width), dtype=torch.float32)
            outputs = torch.zeros(self.width * self.height, dtype=torch.float32)
            return inputs, outputs, -1, -1, 0
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt)
        if img is None:
            # Return black image if gt image doesn't exist
            img = np.zeros((self.height, self.width, 3), dtype=np.float32)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]  # Take first channel only
        img = np.reshape(img, (self.width * self.height))
        return img.astype(np.float32)  # Ensure float32 type
        
    def get_input(self, path1, path2, path3):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img3 = cv2.imread(path3)
        
        if img1 is None or img2 is None or img3 is None:
            raise ValueError(f"Failed to load images: {path1}, {path2}, {path3}")
            
        img1 = cv2.resize(img1, (self.width, self.height))
        img2 = cv2.resize(img2, (self.width, self.height))
        img3 = cv2.resize(img3, (self.width, self.height))
        
        imgs = np.concatenate((img1, img2, img3), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        
        return imgs
