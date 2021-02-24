import lmdb

from create_lmdb_dataset import createDatasetAux, writeCache
import os
import numpy as np
import pathlib
import io

def create_data(rel_in_path, rel_out_path, map_size):
    dir_in = './' + rel_in_path
    dir_out = './' + rel_out_path;
    bg = './bg.jpg';

    os.makedirs(dir_out, exist_ok=True)
    env = lmdb.open(dir_out, map_size=map_size)
    cache = {}
    cnt = 1

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
        cur_chr_path = pathlib.Path(chr_dir_path);
        chr_files = os.listdir(cur_chr_path)
        for chr_file in chr_files:
            cur_chr_img_path = chr_dir_path+'/'+chr_file
            cur_chr_img = Image.open(cur_chr_img_path)
            cur_chr_img = np.asarray(cur_chr_img)
            sz = cur_chr_img.shape
            if sz[2] == 4: # remove the alpha channel
                cur_chr_img = cur_chr_img[:, :, :-1]

            for cur_size in range(0, l_size):
                for r in range(0,cur_size+1):
                    sz = cur_chr_img.shape;
                    background = np.copy(bg[0:sz[0], 0:((cur_size+1)*sz[1]), :])
                    background[:, r * sz[1]:(r + 1) * sz[1], :] = cur_chr_img
                    _, format = chr_file.split('.')
                    imageKey = 'image-%09d'.encode() % cnt
                    labelKey = 'label-%09d'.encode() % cnt
                    background = Image.fromarray(background)
                    with io.BytesIO() as output:
                        background.save(output, format=format)
                        imageBin = output.getvalue()
                    cache[imageKey] = imageBin
                    label = chr(int(chr_dir) + aleph)
                    cache[labelKey] = label.encode()
                    if cnt % 1000 == 0:
                        writeCache(env, cache)
                        cache = {}
                        print('Written %d' % cnt)
                    cnt += 1

                if cur_size == 0:
                    continue

                for c in range(0, cur_size + 1):
                    sz = cur_chr_img.shape;
                    background = np.copy(bg[0:(cur_size+1)*sz[0], 0:sz[1], :])
                    background[c * sz[0]:(c + 1) * sz[0], :, :] = cur_chr_img
                    _, format = chr_file.split('.')
                    imageKey = 'image-%09d'.encode() % cnt
                    labelKey = 'label-%09d'.encode() % cnt
                    background = Image.fromarray(background)
                    with io.BytesIO() as output:
                        background.save(output, format=format)
                        imageBin = output.getvalue()
                    cache[imageKey] = imageBin
                    label = chr(int(chr_dir) + aleph)
                    cache[labelKey] = label.encode()
                    if cnt % 1000 == 0:
                        writeCache(env, cache)
                        cache = {}
                        print('Written %d' % cnt)
                    cnt += 1

                for r in range(0, cur_size + 1):
                    for c in range(0, cur_size + 1):
                        sz = cur_chr_img.shape;
                        background = np.copy(bg[0:((cur_size + 1) * sz[0]), 0:((cur_size + 1) * sz[1]), :])
                        background[c * sz[0]:(c + 1) * sz[0],r * sz[1]:(r + 1) * sz[1], :] = cur_chr_img
                        _, format = chr_file.split('.')
                        imageKey = 'image-%09d'.encode() % cnt
                        labelKey = 'label-%09d'.encode() % cnt
                        background = Image.fromarray(background)
                        with io.BytesIO() as output:
                            background.save(output, format=format)
                            imageBin = output.getvalue()
                        cache[imageKey] = imageBin
                        label = chr(int(chr_dir) + aleph)
                        cache[labelKey] = label.encode()
                        if cnt % 1000 == 0:
                            writeCache(env, cache)
                            cache = {}
                            print('Written %d' % cnt)
                        cnt += 1





if __name__ == '__main__':

    directory = "lmdb"
    parent_dir = "."
    lmdb_path = os.path.join(parent_dir, directory)
    if not os.path.exists(lmdb_path):
        os.mkdir(lmdb_path)
        lmdb_train = os.path.join(str(lmdb_path), 'train')
        os.mkdir(lmdb_train)
        lmdb_val = os.path.join(str(lmdb_path), 'val')
        os.mkdir(lmdb_val)
        #print('creating lmdb for train data')
        #create_data('/hhd_dataset/train/', './lmdb/train/', 4000000000)
        print('creating lmdb for validation data')
        create_data('/hhd_dataset/val/', './lmdb/vals/', 500000000)


