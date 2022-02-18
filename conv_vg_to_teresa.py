import zipfile
import io
import json
import lzma

import numpy as np
import owlready2 as owl

import telenet.dataset_data as tn_data
from telenet.config import get as tn_config

from tqdm import tqdm
from pathlib import Path

DATASET_NAME = tn_config('convert.dataset')
DO_AUGMENTATION = tn_config('convert.augmentation')

def reversemultimap(a):
	for k,v in a.items():
		for p in v: yield (p,k)

with open(tn_data.path(f'{DATASET_NAME}-names.json'), 'rt', encoding='utf-8') as f:
	VG_REL_NAMES = json.load(f)['rels']
	VG_REL_TO_ID = { k:i for i,k in enumerate(VG_REL_NAMES) }

tn_data.load_names('teresa-names.json')
TERESA_TO_VG = tn_config('teresa.predicate_map')
VG_TO_TERESA = { VG_REL_TO_ID[p]:tn_data.CLASS_REL_TO_ID[k] for p,k in reversemultimap(TERESA_TO_VG) }

owl.JAVA_EXE = tn_config('paths.java')
onto = owl.get_ontology(Path(tn_data.path('TeleportaOnto.owl')).as_uri()).load()
ONTO_RELS = [ onto.search_one(label=label) for label in tn_data.REL_NAMES ]

class GenericObject(owl.Thing):
	namespace = onto

def sanitize(a):
	if a is None:
		return []
	if type(a) is list:
		return a
	if type(a) is owl.IndividualValueList:
		return list(a)
	return [a]

def convert_split(split):
	out_split = []
	with_thing = 0; without_thing = 0
	numrels_old = 0; numrels_new = 0
	for img in tqdm(split):
		relmap = {}
		for rel in img['rels']:
			srcdst = (rel['si'], rel['di'])
			for v in rel['v']:
				relid = VG_TO_TERESA.get(v, -1)
				if relid < 0: continue
				pair = relmap.get(srcdst, None)
				if not pair: pair = relmap[srcdst] = set()
				pair.add(relid)
		#print(relmap)
		if len(relmap) != 0:
			with_thing += 1
		else:
			without_thing += 1
			continue

		# Create (dummy) objects
		img_objs = img['objs']
		onto_objs = []
		for i in range(len(img_objs)):
			onto_objs.append(curobj := GenericObject())
			curobj.label = i #curobj.name

		# Insert relations
		for (src,dst),relset in relmap.items():
			for rel in relset:
				with onto:
					owl.default_world.sparql("INSERT { ??1 ??2 ??3 . } WHERE { }", params=(onto_objs[src],ONTO_RELS[rel],onto_objs[dst]))

		# Invoke reasoner
		#with onto: owl.sync_reasoner(debug=False)

		# Retrieve new relations
		onto_obj_to_id = { v:k for k,v in enumerate(onto_objs) }
		newrelmap = {}
		for srcid,src in enumerate(onto_objs):
			for relid,rel in enumerate(ONTO_RELS):
				r1 = sanitize(getattr(src, rel.name))
				if DO_AUGMENTATION:
					r2 = sanitize(getattr(src, 'INDIRECT_' + rel.name))
					tgts = set(r1 + r2)
				else:
					tgts = set(r1)
				if len(tgts) == 0: continue
				#print(src,rel,tgts)
				for tgt in tgts:
					dstid = onto_obj_to_id[tgt]
					#print(srcid,dstid,relid)
					srcdst = (srcid,dstid)
					pair = newrelmap.get(srcdst, None)
					if not pair: pair = newrelmap[srcdst] = set()
					pair.add(relid)

		# Update counters
		numrels_old += sum(len(v) for v in relmap.values())
		numrels_new += sum(len(v) for v in newrelmap.values())

		# Update relation map
		newrellist = img['rels'] = []
		for (src,dst),relset in newrelmap.items():
			newrellist.append({
				'si': src,
				'sv': img_objs[src]['v'],
				'di': dst,
				'dv': img_objs[dst]['v'],
				'v': list(relset)
			})

		# Update split
		out_split.append(img)

		# Cleanup
		for a in onto_objs: owl.destroy_entity(a)
		onto_objs = None

	print('with', with_thing)
	print('without', without_thing)
	print('old numrels', numrels_old)
	print('new numrels', numrels_new)
	return out_split

TERESA_TRAIN = convert_split(tn_data.load_json_xz(f'{DATASET_NAME}-train'))

with lzma.open(f'testdata/{DATASET_NAME}teresa{"" if DO_AUGMENTATION else "noaug"}-train.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(TERESA_TRAIN, f)
