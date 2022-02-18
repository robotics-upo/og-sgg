import tensorflow as tf
import numpy as np
import zipfile
import json
import lzma
import telenet.dataset_data as tn_data
from telenet.utils import load_image_for_vrd_yolo, mdl_yolo, parse_yolo_results
from telenet.config import get as tn_config
from tqdm import tqdm

TERESA_DIR = tn_config('paths.teresa')

IMAGE_LIST = [ img['id'] for img in tn_data.load_json_xz('teresa-test') ]

def load_image(index):
	imgid = IMAGE_LIST[int(index)]
	with open(f"{TERESA_DIR}/handpicked/{imgid}.jpg", 'rb') as f:
		imgdata = f.read()
	img, w, h = load_image_for_vrd_yolo(imgdata)
	return imgid, img, w, h

test_dataset = tf.data.Dataset.from_tensor_slices(list(range(len(IMAGE_LIST)))).map(
	lambda x: tf.py_function(func=load_image, inp=[x], Tout=[tf.string, tf.float32, tf.float32, tf.float32]),
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

convert_dataset(test_dataset, 'teresa-yolo-test.zip', 'teresa-yolo-test-objs.json.xz')
