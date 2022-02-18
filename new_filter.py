import numpy as np
import json
import zipfile
import lzma
import telenet.dataset_data as tn_data
from telenet.config import get as tn_config
from tqdm import tqdm
from itertools import chain

VG_PATH = tn_config('paths.vg')
GQA_PATH = tn_config('paths.gqa')

tn_data.load_names('vg-names.json')

print('Loading VG splits...')
vg_train = tn_data.load_json_xz('vg-train')
vg_test = tn_data.load_json_xz('vg-test')

with zipfile.ZipFile(f'{GQA_PATH}/sceneGraphs.zip', 'r') as zf:
	print('Loading GQA (training split)...')
	with zf.open('train_sceneGraphs.json') as f:
		gqa_train = json.load(f)

	print('Loading GQA (eval split)...')
	with zf.open('val_sceneGraphs.json') as f:
		gqa_eval = json.load(f)

indoor_images = set()

for imid,img in chain(gqa_train.items(), gqa_eval.items()):
	if img.get('location')=='indoors':
		indoor_images.add(imid)

def filter_split(split):
	new_split = []
	for img in tqdm(split):
		if img['id'] in indoor_images:
			new_split.append(img)
	return new_split

print('Filtering splits')
vg_train = filter_split(vg_train)
vg_test = filter_split(vg_test)

oops_train = np.zeros((tn_data.NUM_RELS,), np.bool8)
oops_test = np.zeros((tn_data.NUM_RELS,), np.bool8)

for split,oops in ((vg_train,oops_train),(vg_test,oops_test)):
	for img in split:
		for rel in img['rels']:
			for i in rel['v']:
				oops[i] = True

oops = np.logical_and(oops_train, oops_test)
bad_rels = set()
for i,b in enumerate(oops):
	if not bool(int(b)):
		bad_rels.add(i)
		print(f'"{tn_data.REL_NAMES[i]}" missing')

NEW_RELS = [ (i,rel) for i,rel in enumerate(tn_data.REL_NAMES) if i not in bad_rels ]
OLD_TO_NEW = { old:new for new,(old,_) in enumerate(NEW_RELS) }
NEW_RELS = [ rel for _,rel in NEW_RELS ]

for split in (vg_train,vg_test):
	for img in split:
		newrels = []
		for rel in img['rels']:
			r = set(rel['v'])
			r.difference_update(bad_rels)
			if len(r) == 0:
				continue
			rel['v'] = [ OLD_TO_NEW[q] for q in r ]
			newrels.append(rel)
		img['rels'] = newrels
		if len(newrels) == 0:
			raise 'Have to drop image'

print(f'Training: {len(vg_train)}')
print(f'Testing: {len(vg_test)}')

with open('testdata/vgfilter-names.json', 'wt', encoding='utf-8') as f:
	json.dump({ 'objs': tn_data.CLASS_NAMES, 'attrs': [], 'rels': NEW_RELS }, f)

with lzma.open(f'testdata/vgfilter-train.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(vg_train, f)

with lzma.open(f'testdata/vgfilter-test.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(vg_test, f)
