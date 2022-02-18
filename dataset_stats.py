import numpy as np

import telenet.dataset_data as tn_data

from tqdm import tqdm

DATASET_NAME = 'teresa'
SPLIT_NAMES = [ 'test' ]
#SPLIT_NAMES = [ 'train', 'test' ]
#SPLIT_NAMES = [ 'test' ]

num_images = 0
num_total_objects = 0
num_used_objects = 0
num_total_objpairs = 0
num_objpairs = 0
num_ann_objpairs = 0
num_rels = 0

for splname in SPLIT_NAMES:
	split = tn_data.load_json_xz(f'{DATASET_NAME}-{splname}')
	num_images += len(split)
	for img in split:
		img_usedobj = set()
		for rel in img['rels']:
			num_rels += len(rel['v'])
			img_usedobj.add(rel['si'])
			img_usedobj.add(rel['di'])
		num_total_objects += len(img['objs'])
		num_total_objpairs += len(img['objs'])*(len(img['objs'])-1)
		num_used_objects += len(img_usedobj)
		num_objpairs += len(img_usedobj) * (len(img_usedobj) - 1)
		num_ann_objpairs += len(img['rels'])

print(f'Stats for {DATASET_NAME} dataset ({", ".join(SPLIT_NAMES)})')
print()
print(f'# of images:          {num_images}')
print(f'# of objects:         {num_used_objects}')
print(f'# of extra objects:   {num_total_objects - num_used_objects}')
print(f'# of object pairs:    {num_objpairs}')
print(f'# of extra pairs:     {num_total_objpairs - num_objpairs}')
print(f'# of annotated pairs: {num_ann_objpairs}')
print(f'# of triplets:        {num_rels}')
print()
print(f'Objects/image:        {num_used_objects/num_images:.2f}')
print(f'Triplets/image:       {num_rels/num_images:.2f}')
print(f'Ann. pairs/image:     {num_ann_objpairs/num_images:.2f}')
print(f'Triplets/ann. pair:   {num_rels/num_ann_objpairs:.2f}')
print(f'% of pairs annotated: {num_ann_objpairs*100./num_objpairs:.2f}%')
