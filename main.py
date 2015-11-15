import cv2
import numpy as np
import time
from os import listdir
import os


def match(image_name, template_name):
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    for meth in methods:
        method = eval(meth)
        img = cv2.imread(image_name)
        template = cv2.imread(template_name)
        t = time.time()
        for x in range(100):
            res = cv2.matchTemplate(img,template, method)
        print(meth + ' time of work is {:6.3f} minutes'.format((time.time() - t)))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        h, w, _ = template.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)
        #aaa = np.ndarray(shape=(100, , 3), dtype=np.uint8)
        res = cv2.normalize(res, res, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        t = np.ndarray(shape=(100, res.shape[1]), dtype=np.uint8)
        for index in range(100):
            t[index,:] = res[0,:]
        cv2.imshow('res', t)
        cv2.imshow('img', img)
        cv2.imwrite(meth + '-src.png', img)
        cv2.imwrite(meth + '-mask.png', t)
        #cv2.waitKey()


def create_signature_by_video(video_name, signature_name, type_=0):
    """Type: 0 - each line is every frame
             1 - each line is avg from 30 frames
             3 - each line is 1 frame of 30"""
    cap = cv2.VideoCapture(video_name)
    v = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(v)
    i = 0
    t = time.time()
    if type_ == 0:
        arr = np.ndarray(shape=(h, v, 3), dtype=np.uint8)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for index in range(h):
                arr[index, i] = np.average(frame[index, ], axis=0)
            i += 1
            if not (i % 100):
                print(i)
    elif type_ == 1:
        i2 = 0
        frames_count = 30
        arr = np.ndarray(shape=(h, v/30+1, 3), dtype=np.uint8)
        tmp_images = np.ndarray(shape=(30, h, 3), dtype=np.uint8)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.reduce(frame, 1, 1)
            for index in range(h):
                tmp_images[i % frames_count, index] = img[index]
            if i and i % frames_count == 0:
                arr[:, i // frames_count] = np.average(tmp_images, axis=0)
            i += 1
    print('time of work is {:6.3f} minutes'.format((time.time() - t) / 60))
    cv2.imwrite(signature_name, arr)


def test_with_videocapture(video_name, signature_name):
    cap = cv2.VideoCapture(video_name)
    v = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(v)
    i = 0
    t = time.time()
    i2 = 0
    frames_count = 30
    arr = np.ndarray(shape=(h, v/30+1, 3), dtype=np.uint8)
    tmp_images = np.ndarray(shape=(30, h, 3), dtype=np.uint8)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.reduce(frame, 1, 1)
        for index in range(h):
            tmp_images[i % frames_count, index] = img[index]
        if i and i % frames_count == 0:
            arr[:, i // frames_count] = np.average(tmp_images, axis=0)
        i += 1
    print('time of work is {:6.3f} minutes'.format((time.time() - t) / 60))
    cv2.imwrite(signature_name, arr)

match('signatures/Hraniteli.snov.2012.RUS.BDRip.XviD.AC3.-HQCLUB.png', 'signatures/test1.png')
dir_ = 'cit'
for f in listdir(dir_):
    image_name = 'signatures/' + f[:f.rfind('.')] + ".png"
    print(os.path.join(dir_, f), image_name)
    #create_signature_by_video(os.path.join(dir_, f), image_name, type_=1)
