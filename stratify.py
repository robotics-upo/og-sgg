import json
import lzma
import random

import telenet.dataset_data as tn_data
from telenet.config import get as tn_config

DATASET_NAME = tn_config('stratify.dataset')
SPLIT_SIZE = tn_config('stratify.split_size')

TRAIN_DATA = tn_data.load_json_xz(f'{DATASET_NAME}-train')

rel2img = {}
for idx,img in enumerate(TRAIN_DATA):
	for rel in img['rels']:
		for i in rel['v']:
			s = rel2img.get(i,None)
			if not s:
				rel2img[i] = s = set()
			s.add(idx)

order = list((k,v) for k,v in sorted(rel2img.items(), key=lambda r: len(r[1])))
for i in range(len(rel2img)-1):
	_,imgs = order[0]
	tmp = {}
	for rel,otherimgs in order[1:]:
		otherimgs.difference_update(imgs)
		tmp[rel] = otherimgs
	order = list((k,v) for k,v in sorted(tmp.items(), key=lambda r: len(r[1])))

train_imgs = []
val_imgs = []
for imgset in rel2img.values():
	imgset = list(imgset)
	random.shuffle(imgset)
	cutpoint = round(SPLIT_SIZE*len(imgset))
	val_imgs.extend(imgset[0:cutpoint])
	train_imgs.extend(imgset[cutpoint:])

train_imgs = [ TRAIN_DATA[i] for i in train_imgs ]
val_imgs   = [ TRAIN_DATA[i] for i in val_imgs   ]

with lzma.open(tn_data.path(f'{DATASET_NAME}-train-without-val.json.xz'), 'wt', encoding='utf-8') as f:
	json.dump(train_imgs, f)

with lzma.open(tn_data.path(f'{DATASET_NAME}-val.json.xz'), 'wt', encoding='utf-8') as f:
	json.dump(val_imgs, f)
