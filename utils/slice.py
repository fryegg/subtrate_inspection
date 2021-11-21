import os
from PIL import Image
from itertools import product

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

if __name__=='__main__':
    in_dir = '../../PaDiM-Anomaly-Detection-Localization-master/mvtec_anomaly_detection/4/ground_truth/defect/'
    out_dir = '../4_origin/ground_truth/defect/'
    file_list = os.listdir(in_dir)

    for file_name in file_list:
        tile(file_name, in_dir, out_dir, 1944)