import matplotlib.pyplot as plt
import os
import glob

def load_img_dataset_into_list(dataset_dir):
    img_files = []
    for file in glob.glob(dataset_dir+"*.jpg"):
        img_files.append(file)

    return img_files
