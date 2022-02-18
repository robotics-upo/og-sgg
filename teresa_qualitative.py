import zipfile
import io
import json
import colorsys

import cv2
import numpy as np
import owlready2 as owl

import telenet.dataset_data as tn_data
from telenet.config import get as tn_config

from tqdm import tqdm
from pathlib import Path
from graphviz import Digraph

DATASET_NAME = 'vg'
MODEL_VARIANT = 'VgBaseDec15+QuadroTest'
USE_POST_PROC = False

tn_data.load_names(f'teresa-names.json')
PAIR_CONSTRAINTS = tn_data.load_npy_xz(f'teresa-pair-constraints')
owl.JAVA_EXE = tn_config('paths.java')
onto = owl.get_ontology(Path(tn_data.path('TeleportaOnto.owl')).as_uri()).load()

ONTO_CLS  = [ onto.search_one(label=label.replace(' ','')) for label in tn_data.CLASS_NAMES ]
ONTO_RELS = [ onto.search_one(label=label) for label in tn_data.REL_NAMES ]
ONTO_REL_OBJ_TO_ID = { rel:i for i,rel in enumerate(ONTO_RELS) }
ONTO_INV = [ ONTO_REL_OBJ_TO_ID.get(rel.inverse_property, -1) for rel in ONTO_RELS]
# Note that symmetric predicates get tagged as their own inverses

PREFERS_SUBJECT = tn_config('teresa.qualitative.prefers_subject')
PREFERS_SUBJECT = set(tn_data.CLASS_NAME_TO_ID[x] for x in PREFERS_SUBJECT)
PREFERS_OBJECT  = tn_config('teresa.qualitative.prefers_object')
PREFERS_OBJECT  = set(tn_data.CLASS_NAME_TO_ID[x] for x in PREFERS_OBJECT)

zf_scores = zipfile.ZipFile(f'test-results/teresa+{MODEL_VARIANT}.zip', 'r')
testimgs = tn_data.load_json_xz(f'teresa-test')

if DATASET_NAME != 'teresa':
	with open(tn_data.path(f'{DATASET_NAME}-names.json'), 'rt', encoding='utf-8') as f:
		VG_REL_NAMES = json.load(f)['rels']
		VG_REL_TO_ID = { k:i for i,k in enumerate(VG_REL_NAMES) }

	TERESA_TO_VG = tn_config('teresa.predicate_map')
	TERESA_TO_VG = { tn_data.CLASS_REL_TO_ID[k]:tuple(VG_REL_TO_ID[p] for p in v) for k,v in TERESA_TO_VG.items() }

def color_wheel(i,n):
	return tuple(int(128 + 127*x + .5) for x in colorsys.hsv_to_rgb(float(i)/n, 1., 1.))

def htmlclr(clr):
	return f'#{clr[2]:02x}{clr[1]:02x}{clr[0]:02x}'

def draw_bbox(img, bb, bbox_color, label):
	h,w,_ = img.shape

	bbx,bby,bbw,bbh = bb

	fontScale = 0.5
	bbox_thick = int(0.6 * (w + h) / 600)
	c1, c2 = (bbx, bby), (bbx+bbw, bby+bbh)
	cv2.rectangle(img, c1, c2, bbox_color, bbox_thick)

	t_size = cv2.getTextSize(label, 0, fontScale, thickness=bbox_thick // 2)[0]
	c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
	cv2.rectangle(img, c1, (np.int32(c3[0]), np.int32(c3[1])), bbox_color, -1) #filled

	cv2.putText(img, label,
		(c1[0], np.int32(c1[1] - 2)),
		cv2.FONT_HERSHEY_SIMPLEX,
		fontScale,
		(0, 0, 0),
		bbox_thick // 2,
		lineType=cv2.LINE_AA
	)

def stupid_adapter(f):
	return io.BytesIO(f.read())

def generate_pairs(N):
	for i in range(N):
		for j in range(N):
			if i != j:
				yield (i,j)

def convert_vg_to_teresa(vg_scores):
	t_scores = np.full((tn_data.NUM_RELS,), np.nan)
	for i,tup in TERESA_TO_VG.items():
		scores = np.take(vg_scores, tup)
		scores = scores[np.isfinite(scores)]
		if scores.size != 0:
			t_scores[i] = np.mean(scores)
	return t_scores

def generate_pairs_for_preddet(all_scores, objs):
	num_objs = len(objs)
	if all_scores.shape[0] != num_objs*(num_objs-1):
		print('Bad:', all_scores.shape[0], num_objs, num_objs*(num_objs-1))
		return None
	def generator():
		for i,(src,dst) in enumerate(generate_pairs(num_objs)):
			scores = all_scores[i]
			if DATASET_NAME == 'vg' or DATASET_NAME == 'vgfilter':
				scores = convert_vg_to_teresa(scores)
			elif DATASET_NAME != 'teresa':
				raise Exception("Dunno what to do")
			if USE_POST_PROC:
				constr = PAIR_CONSTRAINTS[objs[src]['v'], objs[dst]['v'], :]
				scores = np.where(constr, scores, np.nan)
			if not np.any(np.isfinite(scores)): continue
			yield (src, dst, scores)
	return generator

for img in tqdm(testimgs):
	id = img['id']
	if len(img['objs']) < 2: continue

	with stupid_adapter(zf_scores.open(f'{id}.npy','r')) as f:
		all_scores = np.load(f)

	pairs = generate_pairs_for_preddet(all_scores, img['objs'])
	if pairs is None:
		print(f'Image with problem: {id}')
		continue

	imgdata = cv2.imread(f'/mnt/data/work/datasets/teresa/handpicked/{id}.jpg')

	objlblcnt = {}
	objnames = []
	objlabels = []
	for obj in img['objs']:
		objcls = obj['v']
		objlblcnt[objcls] = objclsnum = objlblcnt.get(objcls, 0) + 1
		objclsname = tn_data.CLASS_NAMES[objcls].replace(' ','')
		objname = f'{objclsname}_{objclsnum}'
		objnames.append(objname)
		objlabels.append(f'<{objclsname}<SUB>{objclsnum}</SUB>>')
		draw_bbox(imgdata, obj['bb'], color_wheel(len(objlabels)-1, len(img['objs'])), objname)

	cv2.imwrite(f'qualit/{id}_ann.jpg', imgdata)

	triplets = []
	for src,dst,scores in pairs():
		for rel,score in enumerate(scores):
			if np.isfinite(score):
				score = float(score)
				triplets.append((float(score),rel,src,dst))

	g = Digraph()
	g.engine = 'neato'
	g.graph_attr['margin'] = '0'
	g.node_attr['fontsize'] = '12'
	g.node_attr['margin'] = '0'
	g.node_attr['width'] = '0'
	g.node_attr['height'] = '0'
	g.edge_attr['fontsize'] = '9'
	g.edge_attr['arrowhead'] = 'lvee'
	g.edge_attr['arrowtail'] = 'lvee'
	g.edge_attr['len'] = '1.0'
	g.graph_attr['overlap'] = 'scalexy'

	used_objs = set()
	accepted_triplets = set()
	edge_heads = {}
	edge_tails = {}

	def add_triplet(src,dst,rel,is_raw=False):
		cur_triplet = (src,dst,rel)
		cur_edge_head = (src,rel)
		cur_edge_tail = (dst,rel)

		if cur_triplet in accepted_triplets:
			return False # Redundant

		if USE_POST_PROC and not is_raw:
			ontorel = ONTO_RELS[rel]

			if issubclass(ontorel, owl.FunctionalProperty):
				if cur_edge_head in edge_heads:
					return False # Culled

			if issubclass(ontorel, owl.InverseFunctionalProperty):
				if cur_edge_tail in edge_tails:
					return False # Culled

			if (relinv := ONTO_INV[rel]) >= 0: # owl.SymmetricProperty implies ONTO_INV[k] = k
				add_triplet(dst,src,relinv,is_raw=True)
			elif issubclass(ontorel, owl.AsymmetricProperty) and (dst,src,rel) in accepted_triplets:
				return False # Culled

		for obj in (src,dst):
			if obj not in used_objs:
				used_objs.add(obj)
				g.node(f'o{obj}', objlabels[obj], style='filled', color=htmlclr(color_wheel(obj, len(objlabels))))

		accepted_triplets.add(cur_triplet)

		ehset = edge_heads.get(cur_edge_head, None)
		if not ehset: ehset = edge_heads[cur_edge_head] = set()
		ehset.add(dst)

		etset = edge_tails.get(cur_edge_tail, None)
		if not etset: etset = edge_tails[cur_edge_tail] = set()
		etset.add(src)

		return True

	triplets.sort(key=lambda x:x[0], reverse=True)
	num_accepted = 0
	key_triplets = []
	for score,rel,src,dst in triplets:
		if num_accepted >= 16: #(USE_POST_PROC and score <= 0) or
			break
		if add_triplet(src,dst,rel):
			num_accepted += 1
			key_triplets.append((src,dst,rel,score))

	"""
	for src,dst,rel in accepted_triplets:
		attrs = { 'xlabel': tn_data.REL_NAMES[rel] }
		if (dst,src,rel) in accepted_triplets:
			if dst < src:
				continue
			attrs['dir'] = 'both'
		g.edge(f'o{src}', f'o{dst}', **attrs)

	g.render(f'qualit/{id}+{MODEL_VARIANT}+Preproc={"True" if USE_POST_PROC else "False"}.gv')
	"""

	print(f'Image {id}:')
	for src,dst,rel,score in key_triplets:
		if USE_POST_PROC:
			if img['objs'][dst]['v'] in PREFERS_SUBJECT or img['objs'][src]['v'] in PREFERS_OBJECT:
				# Attempt to swap
				if (relinv := ONTO_INV[rel]) >= 0:
					tmp = src; src = dst; dst = tmp
					rel = relinv
		print(f'  ({score:.3f}) {objnames[src]} {tn_data.REL_NAMES[rel]} {objnames[dst]}')
		attrs = { 'xlabel': tn_data.REL_NAMES[rel] }
		if ONTO_INV[rel] == rel:
			attrs['dir'] = 'both'
		g.edge(f'o{src}', f'o{dst}', **attrs)

	g.render(f'qualit/{id}+{MODEL_VARIANT}+Preproc={"True" if USE_POST_PROC else "False"}.gv')
