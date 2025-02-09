import numpy as np
import pandas as pd
import os
import cv2
import argparse

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array = gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height):
    gaussian_kernel_array = create_gaussian(size, variance)
    game = 'game1'  # Assuming we're processing game1 directory
    clips = os.listdir(os.path.join(path_input, game))
    for clip in clips:
        print('clip = {}'.format(clip))

        path_out_game = os.path.join(path_output, game)
        if not os.path.exists(path_out_game):
            os.makedirs(path_out_game)

        path_out_clip = os.path.join(path_out_game, clip)    
        if not os.path.exists(path_out_clip):
            os.makedirs(path_out_clip)  

        path_labels = os.path.join(os.path.join(path_input, game, clip), 'Label.csv')
        labels = pd.read_csv(path_labels)    
        for idx in range(labels.shape[0]):
            file_name, vis, x, y, _ = labels.loc[idx, :]
            heatmap = np.zeros((height, width, 3), dtype=np.uint8)
            if vis in [1, 2]:  
                x = int(float(x)) if not pd.isna(x) else None
                y = int(float(y)) if not pd.isna(y) else None
                if x is not None and y is not None:
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                            if x+i<width and x+i>=0 and y+j<height and y+j>=0:
                                temp = gaussian_kernel_array[i+size][j+size]
                                if temp > 0:
                                    heatmap[y+j,x+i] = (temp,temp,temp)

            cv2.imwrite(os.path.join(path_out_clip, file_name), heatmap)

def create_gt_labels(path_input, path_output, train_rate=0.7):
    df = pd.DataFrame()
    game = 'game1'  # Assuming we're processing game1 directory
    clips = os.listdir(os.path.join(path_input, game))
    
    for clip in clips:
        labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        labels = labels.copy()
        
        # Add new columns
        labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['File Name']
        labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['File Name']
        
        # Create paths for previous frames
        path1_series = labels['path1'].tolist()
        path2_series = path1_series[1:] + [None]  # Add None for last row
        path3_series = path1_series[2:] + [None, None]  # Add None for last two rows
        
        # Create new DataFrame with all required columns
        labels_target = pd.DataFrame({
            'path1': path1_series[2:],
            'path2': path2_series[2:],
            'path3': path3_series[2:],
            'gt_path': labels['gt_path'][2:],
            'X': labels['X'][2:],
            'Y': labels['Y'][2:],
            'Trajectory Pattern': labels['Trajectory Pattern'][2:],
            'Visibility Class': labels['Visibility Class'][2:]
        })
        
        df = pd.concat([df, labels_target], ignore_index=True)
    
    df = df.sample(frac=1)  # Shuffle the data
    num_train = int(df.shape[0]*train_rate)
    df_train = df[:num_train]
    df_test = df[num_train:]
    
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)

if __name__ == '__main__':
    SIZE = 15
    VARIANCE = 8
    WIDTH = 1280
    HEIGHT = 720   

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', type=str, help='path to input folder')
    parser.add_argument('--path_output', type=str, help='path to output folder')
    args = parser.parse_args()
    
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)
        
    create_gt_images(args.path_input, args.path_output, SIZE, VARIANCE, WIDTH, HEIGHT)
    create_gt_labels(args.path_input, args.path_output)

                            
    
    
    
    