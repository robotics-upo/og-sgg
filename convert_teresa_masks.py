import tensorflow_hub as hub
import numpy as np
import zipfile
import telenet.dataset_data as tdsda
from telenet.utils import create_bbox_mask
from tqdm import tqdm

tdsda.load_names('teresa-names.json')

print('Converting semantic vectors...')
mdl_word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
np.save(tdsda.path('teresa-semvecs.npy'), mdl_word2vec(tdsda.CLASS_NAMES))

print('Loading splits...')
teresa_test = tdsda.load_json_xz('teresa-test')

def cnv_bb(obj):
	x,y,w,h = obj['bb']
	return [x,x+w,y,y+h]

def convert_split(split, splitname):
	with zipfile.ZipFile(tdsda.path(f'teresa-mask-{splitname}.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zfo:
		for img in tqdm(split):
			masks = create_bbox_mask(img['w'], img['h'], *(cnv_bb(obj) for obj in img['objs']))
			with zfo.open(f'{img["id"]}.npy','w') as f:
				np.save(f, masks)

convert_split(teresa_test, 'test')
