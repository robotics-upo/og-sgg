import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
#import cv2
import zipfile
import json
import lzma
import os
import telenet.dataset_data as tn_data
from telenet.utils import load_image_for_vrd_yolo, mdl_yolo, parse_yolo_results
from telenet.config import get as tn_config
from tqdm import tqdm

VG_PATH = tn_config('paths.vg')

imgcnvdata = tn_data.load_json_xz('vg-imgcnvdata')
zf1 = zipfile.ZipFile(os.path.join(VG_PATH, 'images.zip'), 'r')
zf2 = zipfile.ZipFile(os.path.join(VG_PATH, 'images2.zip'), 'r')

train_imgs = []
test_imgs = []

for obj in imgcnvdata:
	(train_imgs,test_imgs)[obj['split']].append(obj)

def load_image(db, index):
	obj = db[index]
	if obj['dir'] == 1:
		imgdata = zf1.read(f"VG_100K/{obj['file']}")
	elif obj['dir'] == 2:
		imgdata = zf2.read(f"VG_100K_2/{obj['file']}")
	else:
		raise "Bad dir"
	img, w, h = load_image_for_vrd_yolo(imgdata)
	return obj['id'], img, w, h

def load_train_image(index):
	return load_image(train_imgs, index)

def load_test_image(index):
	return load_image(test_imgs, index)

train_dataset = tf.data.Dataset.from_tensor_slices(list(range(len(train_imgs)))).map(
	lambda x: tf.py_function(func=load_train_image, inp=[x], Tout=[tf.string, tf.float32, tf.float32, tf.float32]),
	num_parallel_calls=tf.data.AUTOTUNE).batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(list(range(len(test_imgs)))).map(
	lambda x: tf.py_function(func=load_test_image, inp=[x], Tout=[tf.string, tf.float32, tf.float32, tf.float32]),
	num_parallel_calls=tf.data.AUTOTUNE).batch(1)

def convert_dataset(dataset, outfile, outfile2):
	res = {}
	with zipfile.ZipFile(tn_data.path(outfile), 'w') as zfo:
		for names,img,widths,heights in tqdm(dataset):
			names = names.numpy()
			features,yolodata = mdl_yolo(img)
			for imnm,imft,imyl,imw,imh in zip(names,features,yolodata,widths,heights):
				imnm = imnm.decode('utf-8')
				res[imnm] = parse_yolo_results(np.expand_dims(imyl, axis=0), imw, imh)
				with zfo.open(f'{imnm}.npy','w') as f:
					np.save(f, imft)
	with lzma.open(tn_data.path(outfile2), 'wt', encoding='utf-8') as f:
		json.dump(res, f)

convert_dataset(train_dataset, 'vg-yolo-train.zip', 'vg-yolo-train-objs.json.xz')
convert_dataset(test_dataset, 'vg-yolo-test.zip', 'vg-yolo-test-objs.json.xz')
