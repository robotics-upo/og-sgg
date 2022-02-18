import json
import lzma
import os
import zipfile

import cv2
import numpy as np
from tqdm import tqdm

from telenet.config import get as tn_config

VRD_PATH = tn_config('paths.vrd')

with zipfile.ZipFile(os.path.join(VRD_PATH, 'vrd_json_dataset.zip'), 'r') as zf:
	print('Loading VRD object names...')
	with zf.open('objects.json') as f:
		vrd_objnames = json.load(f)

	print('Loading VRD relation names...')
	with zf.open('predicates.json') as f:
		vrd_relnames = json.load(f)

	print('Loading VRD (train split)...')
	with zf.open('annotations_train.json') as f:
		vrd_train = json.load(f)

	print('Loading VRD (test split)...')
	with zf.open('annotations_test.json') as f:
		vrd_test = json.load(f)

zfimg = zipfile.ZipFile(os.path.join(VRD_PATH, 'sg_dataset.zip'), 'r')

def conv_vrd_bb(vrdbb):
	ymin,ymax,xmin,xmax = vrdbb
	return [ xmin, ymin, xmax-xmin, ymax-ymin ]

def collect_vrd_info(data):
	objs = []
	objcache = {}
	def get_obj_id(obj):
		key = (obj['category'], obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3])
		id = objcache.get(key, None)
		if id is None:
			id = len(objs)
			objcache[key] = id
			objs.append({
				'v': obj['category'],
				'bb': conv_vrd_bb(obj['bbox'])
			})
		return id

	relmap = {}
	for relobj in data:
		src = get_obj_id(relobj["subject"])
		dst = get_obj_id(relobj["object"])
		rel = relobj["predicate"]
		if src == dst:
			print('WARNING:', vrd_objnames[relobj["subject"]["category"]], vrd_relnames[rel], 'itself')
			continue
		relset = relmap.get((src,dst), None)
		if not relset:
			relmap[(src,dst)] = relset = set()
		relset.add(rel)

	rels = []
	for (src,dst),relset in relmap.items():
		rels.append({
			"n": len(relset),
			"si": src,
			"sv": objs[src]['v'],
			"di": dst,
			"dv": objs[dst]['v'],
			"v":  list(relset)
		})

	if len(rels) == 0:
		return None

	return {
		'objs': objs,
		'rels': rels
	}

def process_vrd(vrd_split, splitname):
	bad = 0
	splitdata = []
	for imgname,data in tqdm(vrd_split.items()):
		data = collect_vrd_info(data)
		if not data:
			print(f'BAD: {imgname} has no relations, skipping')
			bad += 1
			continue

		try:
			imgraw = zfimg.read(f'sg_dataset/sg_{splitname}_images/{imgname}')
		except:
			print(f'BAD: {imgname} cannot be opened, skipping')
			bad += 1
			continue

		imgraw = cv2.imdecode(np.frombuffer(imgraw, np.uint8), cv2.IMREAD_ANYCOLOR)
		h,w,_ = imgraw.shape
		data.update({
			"id": imgname[:imgname.rfind('.')],
			"w": w,
			"h": h
		})
		splitdata.append(data)

	with lzma.open(f'testdata/vrd-{splitname}.json.xz', 'wt', encoding='utf-8') as f:
		json.dump(splitdata, f)

	print(f'{bad} images dropped in {splitname} split')

with open('testdata/vrd-names.json', 'wt', encoding='utf-8') as f:
	json.dump({ 'objs': vrd_objnames, 'attrs': [], 'rels': vrd_relnames }, f)

process_vrd(vrd_train, 'train')
process_vrd(vrd_test, 'test')
