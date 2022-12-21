import tensorflow_hub as hub
import numpy as np
import zipfile
import re
import telenet.dataset_data as tdsda
from telenet.utils import create_bbox_mask
from tqdm import tqdm

q1 = re.compile('([a-z])([A-Z])')
q2 = re.compile('([A-Z])([A-Z][a-z])')

tdsda.load_names('ai2thor-names.json')

tdsda.CLASS_NAMES = [ q1.sub(r'\1 \2',a) for a in tdsda.CLASS_NAMES ]
tdsda.CLASS_NAMES = [ q2.sub(r'\1 \2',a) for a in tdsda.CLASS_NAMES ]
tdsda.CLASS_NAMES = [ a.lower()          for a in tdsda.CLASS_NAMES ]

print('Converting semantic vectors...')
mdl_word2vec = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
np.save(tdsda.path('ai2thor-semvecs.npy'), mdl_word2vec(tdsda.CLASS_NAMES))

print('Loading splits...')
ai2thor_test = tdsda.load_json_xz('ai2thor-test')

def cnv_bb(obj):
	x,y,w,h = obj['bb']
	return [x,x+w,y,y+h]

def convert_split(split, splitname):
	with zipfile.ZipFile(tdsda.path(f'ai2thor-mask-{splitname}.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as zfo:
		for img in tqdm(split):
			masks = create_bbox_mask(img['w'], img['h'], *(cnv_bb(obj) for obj in img['objs']))
			with zfo.open(f'{img["id"]}.npy','w') as f:
				np.save(f, masks)

convert_split(ai2thor_test, 'test')
