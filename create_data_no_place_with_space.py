from collections import defaultdict

import lmdb

from create_lmdb_dataset import createDatasetAux, writeCache
import os
import numpy as np
import pathlib
import io
from PIL import Image, ImageShow
from math import ceil


def def_value():
    return []

def create_words_data_spaces(rel_in_path, rel_out_path, map_size, format, images_num):
    dir_in = './' + rel_in_path
    dir_out = './' + rel_out_path;

    output_format = format

    os.makedirs(dir_out, exist_ok=True)
    env = lmdb.open(dir_out, map_size=map_size)
    cache = {}
    cnt = 1

    dir_in_path = pathlib.Path(dir_in);
    chr_dirs = os.listdir(dir_in_path)
    idx_to_chr_path = defaultdict(def_value)
    aleph = ord('א')
    for chr_dir in chr_dirs:
        #print(chr(int(chr_dir) + aleph))
        chr_dir_path = dir_in + chr_dir
        cur_chr_path = pathlib.Path(chr_dir_path);
        chr_files = os.listdir(cur_chr_path)
        for chr_file in chr_files:
            cur_chr_img_path = chr_dir_path+'/'+chr_file
            idx_to_chr_path[int(chr_dir)].append(cur_chr_img_path)


    non_ending_letters = 'אבגדהוזחטיכלמנסעפצקרשת'
    ending_letters = 'אבגדהוזחטיךלמןסעףץקרשת'
    low_letters='נקךןת'

    bg = './bg.jpg';
    bg = Image.open(bg)
    bg = np.asarray(bg)
    total_bytes = 0
    while cnt-1 < images_num:
        one_or_two = np.random.randint(0, 10)
        if one_or_two == 3:
            word_len = np.random.randint(11, 20)
        else:
            word_len = np.random.randint(1, 11)

        is_lower = np.zeros(word_len)

        images = []
        length_upper = 0
        length_lower = 0
        width = 0

        label = ''
        for i in range(word_len-1):
            letter = np.random.randint(22)
            if non_ending_letters[letter] in low_letters:
                is_lower[word_len-1-i] = 1
            letter_idx = ord(non_ending_letters[letter]) - ord('א')
            label += chr(letter_idx+ord('א'))
            letter_imgs = idx_to_chr_path[letter_idx]
            amount = len(letter_imgs)
            img_num = np.random.randint(amount-1)
            letter_img_path = letter_imgs[img_num]

            cur_chr_img = Image.open(letter_img_path)
            cur_chr_img = np.asarray(cur_chr_img)
            cur_length, cur_width, channels = cur_chr_img.shape
            if channels == 4:  # remove the alpha channel
                cur_chr_img = cur_chr_img[:, :, :-1]
            width += cur_width
            images.append(cur_chr_img)
            if non_ending_letters[letter] in low_letters:
                length_lower = max(length_lower, cur_length)
            else:
                length_upper = max(length_upper, cur_length)

        #last letter in ending letters
        letter = np.random.randint(22)
        if ending_letters[letter] in low_letters:
            is_lower[0] = 1
        letter_idx = ord(ending_letters[letter]) - ord('א')
        label += chr(letter_idx + ord('א'))
        letter_imgs = idx_to_chr_path[letter_idx]
        amount = len(letter_imgs)
        img_num = np.random.randint(amount - 1)
        letter_img_path = letter_imgs[img_num]

        cur_chr_img = Image.open(letter_img_path)
        cur_chr_img = np.asarray(cur_chr_img)
        cur_length, cur_width, channels = cur_chr_img.shape
        if channels == 4:  # remove the alpha channel
            cur_chr_img = cur_chr_img[:, :, :-1]
        width += cur_width
        images.append(cur_chr_img)
        if non_ending_letters[letter] in low_letters:
            length_lower = max(length_lower, cur_length)
        else:
            length_upper = max(length_upper, cur_length)

        if length_upper > length_lower:
            total_length = ceil(length_upper + length_lower/2)
        else:
            total_length = ceil(length_lower + length_upper / 2)
        background = np.copy(bg[0:total_length, 0:2*width, :])

        #word generation
        #print(label)

        width_start_idx = 0
        for i, (letter_img, is_low) in enumerate(zip(reversed(images), is_lower)):
            cur_len, cur_width, _ = letter_img.shape
            if not is_low:
                start_idx = length_upper - cur_len
            else:
                start_idx = total_length - cur_len
            col_end_idx = width_start_idx+cur_width

            background[start_idx:(start_idx+cur_len),
                width_start_idx:(col_end_idx), :] = letter_img
            cur_space_width = cur_width / 10
            cur_space = int(np.random.randint(-2,4) * cur_space_width)
            if i < word_len - 1:
                width_start_idx += cur_width + cur_space
            else:
                width_start_idx += cur_width
                background =  background[:, 0:width_start_idx, :]

        background = Image.fromarray(background)
        ImageShow.show(background)
        #print()
        #save image in lmdb
        #_, format = chr_file.split('.')
        #if output_format == 'same':
            #output_format = format
        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        total_bytes += len(imageKey) + len(labelKey)
        background = Image.fromarray(background)
        with io.BytesIO() as output:
            background.save(output, format=output_format)
            imageBin = output.getvalue()
            total_bytes += len(imageBin) + len(label.encode())
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if cnt % 1000 == 0:
            #print(total_bytes)
            env.set_mapsize(int(total_bytes*1.17))
            writeCache(env, cache)
            cache = {}
            print('Written %d' % cnt)
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples of size %d bytes' % (nSamples, total_bytes))


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
        #create_data('/hhd_dataset/train/', './lmdb/train/', 4000000000, 'jpeg')
        #print('creating lmdb for validation data')
        #create_data('/hhd_dataset/val/', './lmdb/val/', 1500000000, 'jpeg')
        # print('creating lmdb for train data with words')
        print('creating lmdb for train data')
        create_words_data_spaces('/hhd_dataset/train/', './lmdb/train/', 1, 'jpeg', 1000000)
        print('creating lmdb for validation data')
        create_words_data_spaces('/hhd_dataset/val/', './lmdb/val/', 1, 'jpeg', 100000)


