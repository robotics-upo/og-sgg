import tensorflow_hub as hub
import numpy as np
import os
import cv2
import zipfile
import telenet.dataset_data as tdsda
from telenet.utils import create_bbox_mask
from telenet.config import get as tn_config
from tqdm import tqdm

VG_PATH = tn_config('paths.vg')
tdsda.load_names('vg-names.json')

print('Converting semantic vectors...')
mdl_word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
np.save(tdsda.path('vg-semvecs.npy'), mdl_word2vec(tdsda.CLASS_NAMES))

print('Loading splits...')
vg_train = tdsda.load_json_xz('vg-train')
vg_test = tdsda.load_json_xz('vg-test')

def cnv_bb(obj):
	x,y,w,h = obj['bb']
	return [x,x+w,y,y+h]

def convert_split(split, splitname):
	with zipfile.ZipFile(tdsda.path(f'vg-mask-{splitname}.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zfo:
		for img in tqdm(split):
			masks = create_bbox_mask(img['w'], img['h'], *(cnv_bb(obj) for obj in img['objs']))
			with zfo.open(f'{img["id"]}.npy','w') as f:
				np.save(f, masks)

convert_split(vg_train, 'train')
convert_split(vg_test, 'test')
