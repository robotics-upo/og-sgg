import h5py
import numpy as np
import json
import lzma
from tqdm import tqdm

from telenet.config import get as tn_config

VG_PATH = tn_config('paths.vg')

def minmax(x, vmin, vmax):
	return min(max(int(x), vmin), vmax)

with open(f'{VG_PATH}/VG-SGG-dicts.json', 'rt', encoding='utf-8') as f:
	vg_sgg_dicts = json.load(f)

vg_objnames = [""] * max(int(x) for x in vg_sgg_dicts['idx_to_label'].keys())
vg_relnames = [""] * max(int(x) for x in vg_sgg_dicts['idx_to_predicate'].keys())

for k,v in vg_sgg_dicts['idx_to_label'].items():
	vg_objnames[int(k)-1] = v

for k,v in vg_sgg_dicts['idx_to_predicate'].items():
	vg_relnames[int(k)-1] = v

with open(f'{VG_PATH}/image_data.json', 'rt', encoding='utf-8') as f:
	vg_imgmeta = json.load(f)

#print(len(vg_imgmeta))
#assert len(vg_imgmeta) == 108073

vg_images = []
vg_imgurls = []
for obj in vg_imgmeta:
	# Ignore corrupted images
	if int(obj['image_id']) in [1592, 1722, 4616, 4617]:
		continue

	vg_imgurls.append(obj['url'])
	vg_images.append({
		'w': int(obj['width']),
		'h': int(obj['height'])
	})

f = h5py.File(f'{VG_PATH}/VG-SGG.h5', 'r')

"""
<HDF5 dataset "active_object_mask": shape (1145398, 1), type "|b1">
<HDF5 dataset "boxes_1024": shape (1145398, 4), type "<i4">
<HDF5 dataset "boxes_512": shape (1145398, 4), type "<i4">
<HDF5 dataset "img_to_first_box": shape (108073,), type "<i4">
<HDF5 dataset "img_to_first_rel": shape (108073,), type "<i4">
<HDF5 dataset "img_to_last_box": shape (108073,), type "<i4">
<HDF5 dataset "img_to_last_rel": shape (108073,), type "<i4">
<HDF5 dataset "labels": shape (1145398, 1), type "<i8">
<HDF5 dataset "predicates": shape (622705, 1), type "<i8">
<HDF5 dataset "relationships": shape (622705, 2), type "<i4">
<HDF5 dataset "split": shape (108073,), type "<i4">
"""

ds_split = f['split']
ds_img_to_first_box = f['img_to_first_box']
ds_img_to_first_rel = f['img_to_first_rel']
ds_img_to_last_box = f['img_to_last_box']
ds_img_to_last_rel = f['img_to_last_rel']

ds_boxes_1024 = f['boxes_1024']
ds_labels = f['labels']
ds_predicates = f['predicates']
ds_relationships = f['relationships']

assert ds_split.shape[0] == len(vg_images)

vg_train = []
vg_test = []
vg_imgcnvdata = []

for i in tqdm(range(ds_split.shape[0])):
	img = vg_images[i]
	first_box = ds_img_to_first_box[i]
	first_rel = ds_img_to_first_rel[i]
	last_box = ds_img_to_last_box[i]
	last_rel = ds_img_to_last_rel[i]

	if first_box < 0 or first_rel < 0:
		continue

	img_w = img['w']
	img_h = img['h']
	scale = float(max(img_w, img_h)) / 1024.

	# Input box format: x_center, y_center, width, height (scaled so that largest image dimension is 1024)
	# Desired format: x_left, y_top, x_right, y_bottom (using original image dimensions)
	box_coords = ds_boxes_1024[first_box:last_box+1,:].astype(np.float32) * scale
	box_coords[:,:2] -= .5 * box_coords[:,2:]
	box_coords[:,2:] += box_coords[:,:2]
	box_coords = (box_coords + .5).astype(np.int32)

	box_labels = ds_labels[first_box:last_box+1,:]
	rel_preds = ds_predicates[first_rel:last_rel+1,:]
	rel_pairs = ds_relationships[first_rel:last_rel+1,:] - first_box

	img['objs'] = objs = []
	img['rels'] = rels = []

	for bb,clid in zip(box_coords,box_labels):
		clid = int(clid)-1
		if clid < 0 or clid >= len(vg_objnames):
			print('Bad obj')
			clid = 0 # can't skip this without breaking the relation data so let's just do this instead

		x1,y1,x2,y2 = bb
		x1 = minmax(x1, 0, img_w)
		y1 = minmax(y1, 0, img_h)
		x2 = minmax(x2, 0, img_w)
		y2 = minmax(y2, 0, img_h)

		objs.append({
			'v': clid,
			'bb': [ x1, y1, x2-x1, y2-y1 ]
		})

	relmap = {}
	for (s1,s2),p in zip(rel_pairs,rel_preds):
		s1 = int(s1)
		s2 = int(s2)
		p = int(p[0])-1

		if s1 < 0 or s1 >= len(objs):
			print('Bad src obj')
			continue

		if s2 < 0 or s2 >= len(objs):
			print('Bad dst obj')
			continue

		if p < 0 or p >= len(vg_relnames):
			print('Bad rel')
			continue

		q = relmap.get((s1,s2), None)
		if q is None:
			q = relmap[(s1,s2)] = set()
		q.add(p)

	if len(relmap) == 0:
		continue

	for (s1,s2),relset in relmap.items():
		rels.append({
			'si': s1,
			'sv': objs[s1]['v'],
			'di': s2,
			'dv': objs[s2]['v'],
			'v': list(relset)
		})

	split = int(ds_split[i]==2)
	(vg_train,vg_test)[split].append(img)

	url = vg_imgurls[i]
	fname = url[url.rfind('/')+1:]
	img['id'] = fname[:fname.rfind('.')]
	vg_imgcnvdata.append({
		'id': img['id'],
		'file': fname,
		'dir': 2 if '_2/' in url else 1,
		'split': split
	})

print(f'{len(vg_train)} images in training split')
print(f'{len(vg_test)} images in test split')
print(f'{ds_split.shape[0]-len(vg_train)-len(vg_test)} images dropped due to errors or missing data')

with open('testdata/vg-names.json', 'wt', encoding='utf-8') as f:
	json.dump({ 'objs': vg_objnames, 'attrs': [], 'rels': vg_relnames }, f)

with lzma.open('testdata/vg-imgcnvdata.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(vg_imgcnvdata, f)

with lzma.open(f'testdata/vg-train.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(vg_train, f)

with lzma.open(f'testdata/vg-test.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(vg_test, f)
