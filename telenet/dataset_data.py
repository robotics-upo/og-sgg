import numpy as np
import json
import lzma
import os

DATA_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../testdata')

CLASS_NAMES = ATTR_NAMES = REL_NAMES = None
NUM_CLASSES = NUM_ATTRS = NUM_RELS = 0
CLASS_NAME_TO_ID = CLASS_ATTR_TO_ID = CLASS_REL_TO_ID = {}

def path(name):
	return os.path.join(DATA_DIR, name)

def load_names(name):
	global CLASS_NAMES, ATTR_NAMES, REL_NAMES
	global NUM_CLASSES, NUM_ATTRS, NUM_RELS
	global CLASS_NAME_TO_ID, CLASS_ATTR_TO_ID, CLASS_REL_TO_ID

	with open(path(name), 'rt', encoding='utf-8') as f:
		d_names = json.load(f)

	CLASS_NAMES = d_names['objs']
	ATTR_NAMES = d_names['attrs']
	REL_NAMES = d_names['rels']

	NUM_CLASSES = len(CLASS_NAMES)
	NUM_ATTRS = len(ATTR_NAMES)
	NUM_RELS = len(REL_NAMES)

	CLASS_NAME_TO_ID = { v:k for k,v in enumerate(CLASS_NAMES) }
	CLASS_ATTR_TO_ID = { v:k for k,v in enumerate(ATTR_NAMES) }
	CLASS_REL_TO_ID = { v:k for k,v in enumerate(REL_NAMES) }

def load_json_xz(name):
	with lzma.open(path(name + '.json.xz'), 'rt', encoding='utf-8') as f:
		return json.load(f)

def load_npy_xz(name):
	with lzma.open(path(name + '.npy.xz'), 'rb') as f:
		return np.load(f)

def load_priors(name):
	priors = {}
	for pr in load_json_xz(name):
		src = pr['s']
		dst = pr['d']
		vec = np.array(pr['p'], np.float32)
		vec /= np.sum(vec)
		priors[(src,dst)] = vec
	return priors

def load_used_images():
	with open(path('used_images.txt'), 'rt', encoding='utf-8') as f:
		return [int(x) for x in f.read().split('\n')]

with open(path('coco-names.txt'), 'rt', encoding='utf-8') as f:
	COCO_NAMES = f.read().rstrip('\n').split('\n')

COCO_SEMVECS = np.load(path('coco-semvecs.npy'))

