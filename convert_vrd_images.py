import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
#import cv2
import zipfile
import json
import lzma
import os
import telenet.dataset_data as tdsda
from telenet.utils import load_image_for_vrd_yolo, mdl_yolo, parse_yolo_results
from telenet.config import get as tn_config
from tqdm import tqdm

VRD_PATH = tn_config('paths.vrd')

zf = zipfile.ZipFile(os.path.join(VRD_PATH, 'sg_dataset.zip'), 'r')

train_imgs = []
test_imgs = []

def is_image(name):
	name = name[-4:]
	return name == '.jpg' or name == '.gif' or name == '.png'

for name in zf.namelist():
	if not is_image(name):
		continue
	basename = name[name.rfind('/')+1:-4]
	if '_train_' in name:
		train_imgs.append((basename, name))
	elif '_test_' in name:
		test_imgs.append((basename, name))

def load_image(db, index):
	img, w, h = load_image_for_vrd_yolo(zf.read(db[index][1]))
	return db[index][0], img, w, h

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
	with zipfile.ZipFile(tdsda.path(outfile), 'w') as zfo:
		for names,img,widths,heights in tqdm(dataset):
			names = names.numpy()
			features,yolodata = mdl_yolo(img)
			for imnm,imft,imyl,imw,imh in zip(names,features,yolodata,widths,heights):
				imnm = imnm.decode('utf-8')
				res[imnm] = parse_yolo_results(np.expand_dims(imyl, axis=0), imw, imh)
				with zfo.open(f'{imnm}.npy','w') as f:
					np.save(f, imft)
	with lzma.open(outfile2, 'wt', encoding='utf-8') as f:
		json.dump(res, f)

convert_dataset(train_dataset, 'vrd-yolo-train.zip', 'vrd-yolo-train-objs.json.xz')
convert_dataset(test_dataset, 'vrd-yolo-test.zip', 'vrd-yolo-test-objs.json.xz')
