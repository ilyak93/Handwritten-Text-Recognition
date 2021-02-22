
from create_lmdb_dataset import createDatasetAux
import os
import numpy as np
import pathlib

def create_data(rel_in_path, rel_out_path):
    dir_in = './' + rel_in_path
    dir_out = './' + rel_out_path;
    bg = './bg.jpg';

    from PIL import Image
    # Open the image form working directory
    bg = Image.open(bg)
    bg = np.asarray(bg)

    l_size = 3;

    dir_in_path = pathlib.Path(dir_in);

    chr_dirs = os.listdir(dir_in_path)

    aleph = ord('א')
    for chr_dir in chr_dirs:
        print(chr(int(chr_dir) + aleph))
        chr_dir_path = dir_in + chr_dir
        chr_output_dir_path = dir_out + chr_dir
        os.mkdir(chr_output_dir_path);
        cur_chr_path = pathlib.Path(chr_dir_path);
        chr_files = os.listdir(cur_chr_path)
        for chr_file in chr_files:
            cur_chr_img_path = chr_dir_path+'/'+chr_file
            cur_chr_img = Image.open(cur_chr_img_path)
            cur_chr_img = np.asarray(cur_chr_img)
            for cur_size in range(0, l_size):
                for r in range(0,cur_size+1):
                    sz = cur_chr_img.shape;
                    background = np.copy(bg[0:sz[0], 0:((cur_size+1)*sz[1]), :])
                    background[:, r * sz[1]:(r + 1) * sz[1], :] = cur_chr_img
                    (name, format) = chr_file.split('.')
                    Image.fromarray(background).save(
                        dir_out + chr_dir + '/' + 'r'+ name + '_' + str(cur_size) + '_' + str(r)+'.'+format)

                if cur_size == 0:
                    continue

                for c in range(0, cur_size + 1):
                    sz = cur_chr_img.shape;
                    background = np.copy(bg[0:(cur_size+1)*sz[0], 0:sz[1], :])
                    background[c * sz[0]:(c + 1) * sz[0], :, :] = cur_chr_img
                    (name, format) = chr_file.split('.')
                    Image.fromarray(background).save(
                        dir_out + chr_dir + '/' + 'c' + name + '_' + str(cur_size) + '_' + str(c) + '.' + format)

                for r in range(0, cur_size + 1):
                    for c in range(0, cur_size + 1):
                        sz = cur_chr_img.shape;
                        background = np.copy(bg[0:((cur_size + 1) * sz[0]), 0:((cur_size + 1) * sz[1]), :])
                        background[c * sz[0]:(c + 1) * sz[0],r * sz[1]:(r + 1) * sz[1], :] = cur_chr_img
                        (name, format) = chr_file.split('.')
                        Image.fromarray(background).save(
                            dir_out + chr_dir + '/' + 'r_c' + name + '_' + str(cur_size) + '_' + str(r) + '_' + str(c) + '.' + format)







if __name__ == '__main__':

    directory = "expanded_hhd"
    parent_dir = "."
    expanded_path = os.path.join(parent_dir, directory)
    if not os.path.exists(expanded_path):
        os.mkdir(expanded_path)
        train_path = os.path.join(expanded_path, 'train')
        os.mkdir(train_path)
        val_path = os.path.join(expanded_path, 'val')
        os.mkdir(val_path)
        print('creating expanded hhd of training data')
        create_data('/hhd_dataset/train/', '/expanded_hhd/train/')
        print('creating expanded hhd of validation data')
        create_data('/hhd_dataset/val/', '/expanded_hhd/val/')

    directory = "lmdb"
    parent_dir = "."
    lmdb_path = os.path.join(parent_dir, directory)
    if not os.path.exists(lmdb_path):
        os.mkdir(lmdb_path)
        lmdb_train = os.path.join(str(lmdb_path), 'train')
        os.mkdir(lmdb_train)
        lmdb_val = os.path.join(str(lmdb_path), 'val')
        os.mkdir(lmdb_val)
        print('creating lmdb for train data')
        createDatasetAux('./expanded_hhd/train/', './lmdb/train/')
        print('creating lmdb for validation data')
        createDatasetAux('./expanded_hhd/val/', './lmdb/val/')


