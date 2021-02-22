""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=4000000000)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

import pathlib

if __name__ == '__main__':
    #fire.Fire(createDataset)
    with open("G:/data/gt.txt", "w", encoding="utf-8") as gt_file:

        G = pathlib.Path('G:/')
        data = 'data/val'
        data_path = G / data
        all_letter_dirs = os.listdir(data_path)
        aleph = ord('א')
        all_letter_dirs = sorted([int(i) for i in all_letter_dirs])
        for letter_num_dir in all_letter_dirs:
            letter_path = data_path / str(letter_num_dir)
            files =  os.listdir(letter_path)
            for file in files:
                hebrew_char = chr(letter_num_dir+aleph)
                file_path = (letter_path / file).relative_to(G/'data')
                gt_file.write(str(file_path)+'\t'+hebrew_char+'\n')

    createDataset('G:/data', 'G:/data/gt.txt', 'G:/data/val')



def createDatasetAux(path, output_path):
    import pathlib
    input_path = pathlib.Path(path)
    name = input_path.name
    parent = input_path.parents[0]
    gt_name = name+'-gt.txt'
    gt_path = str(parent / gt_name)
    with open(gt_path, "w", encoding="utf-8") as gt_file:
        data_path = pathlib.Path(path)
        all_letter_dirs = os.listdir(data_path)
        aleph = ord('א')
        all_letter_dirs = sorted([int(i) for i in all_letter_dirs])
        for letter_num_dir in all_letter_dirs:
            letter_path = data_path / str(letter_num_dir)
            files =  os.listdir(letter_path)
            for file in files:
                hebrew_char = chr(letter_num_dir+aleph)
                file_path = (letter_path / file).relative_to(input_path)
                gt_file.write(str(file_path)+'\t'+hebrew_char+'\n')

    createDataset(path, gt_path, output_path)

