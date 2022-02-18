import tensorflow_hub as hub
import numpy as np
import os
import cv2
import zipfile
import telenet.dataset_data as tdsda
from telenet.utils import create_bbox_depth_mask
from telenet.config import get as tn_config
from tqdm import tqdm

VRD_PATH = tn_config('paths.vrd')
tdsda.load_names('vrd-names.json')

print('Converting semantic vectors...')
mdl_word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
np.save(tdsda.path('vrd-semvecs.npy'), mdl_word2vec(tdsda.CLASS_NAMES))

print('Loading splits...')
vrd_train = tdsda.load_json_xz('vrd-train')
vrd_test = tdsda.load_json_xz('vrd-test')

zfdepth = zipfile.ZipFile(os.path.join(VRD_PATH, 'vrd-depth.zip'), 'r')

def cnv_bb(obj):
	x,y,w,h = obj['bb']
	return [x,x+w,y,y+h]

def convert_split(vrd_split, splitname):
	with zipfile.ZipFile(tdsda.path(f'vrd-mask-{splitname}.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zfo:
		for img in tqdm(vrd_split):
			depth_img = cv2.imdecode(np.frombuffer(zfdepth.read(f'{splitname}/{img["id"]}.png'), np.uint8), cv2.IMREAD_ANYDEPTH).astype('float')
			mmax = np.max(depth_img)
			mmin = np.min(depth_img)
			depth_img = 1. - (depth_img - mmin) / (mmax - mmin)
			depth_img = np.expand_dims(depth_img, 2)
			masks = create_bbox_depth_mask(depth_img, *(cnv_bb(obj) for obj in img['objs']))
			with zfo.open(f'{img["id"]}.npy','w') as f:
				np.save(f, masks)

convert_split(vrd_train, 'train')
convert_split(vrd_test, 'test')
