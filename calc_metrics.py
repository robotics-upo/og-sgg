import zipfile
import io

import numpy as np

import telenet.dataset_data as tn_data
from telenet.config import get as tn_config

from tqdm import tqdm

DATASET_NAME = tn_config('test.dataset')
MODEL_VARIANT = tn_config('model.variant')

tn_data.load_names(f'{DATASET_NAME}-names.json')
zf_scores = zipfile.ZipFile(f'test-results/{MODEL_VARIANT}.zip', 'r')
trainimgs = tn_data.load_json_xz(f'{DATASET_NAME}-train')
testimgs = tn_data.load_json_xz(f'{DATASET_NAME}-test')

def stupid_adapter(f):
	return io.BytesIO(f.read())

def generate_pairs(N):
	for i in range(N):
		for j in range(N):
			if i != j:
				yield (i,j)

def generate_pairs_for_preddet(all_scores, num_objs):
	if all_scores.shape[0] != num_objs*(num_objs-1):
		print('Bad:', all_scores.shape[0], num_objs, num_objs*(num_objs-1))
		return None
	def generator():
		for i,(src,dst) in enumerate(generate_pairs(num_objs)):
			yield (src, dst, all_scores[i])
	return generator

def count_rels(ground_truth):
	rels = {}
	for _,_,relid in ground_truth:
		rels[relid] = rels.get(relid,0) + 1
	return rels

def extract_scores(ground_truth, annotated_pairs, pairgen, cutoffs=[1,tn_data.NUM_RELS]):
	cutoffs = set(cutoffs)
	matches = { k:[] for k in cutoffs }
	scores  = { k:[] for k in cutoffs }
	relids  = { k:[] for k in cutoffs }
	for src,dst,scorevec in pairgen():
		#if (src,dst) not in annotated_pairs:
		#	continue
		order = np.argsort(-scorevec)
		for p in range(tn_data.NUM_RELS):
			relid = order[p]
			score = scorevec[relid]
			match = 1 if (src,dst,relid) in ground_truth else 0
			if not np.isfinite(score):
				continue
			for k in cutoffs:
				if p < k:
					matches[k].append(match)
					scores[k].append(score)
					relids[k].append(relid)
	for k in cutoffs:
		matches[k] = np.array(matches[k])
		scores[k] = np.array(scores[k])
		relids[k] = np.array(relids[k])
	return matches, scores, relids

class RecallAggregator:
	def __init__(self):
		self.accum = 0.
		self.num_images = 0
		self.num_matches = 0
		self.num_gtrels = 0

	def update(self, matches, gt):
		assert gt != 0
		self.accum += float(matches) / float(gt)
		self.num_images += 1
		self.num_matches += matches
		self.num_gtrels += gt

	def result(self):
		return (self.accum / self.num_images, self.num_matches / self.num_gtrels)

def update_recall(recall, mean_recall, ground_truth, matches, scores, relids, values=[20,50,100]):
	relcnt = count_rels(ground_truth)
	GT = len(ground_truth)
	values = set(values)
	for k,matches_K in matches.items():
		scores_K = scores[k]
		relids_K = relids[k]
		for RK in values:
			RK_k = (RK,k)

			recall_RK_k = recall.get(RK_k, None)
			if not recall_RK_k:
				recall_RK_k = recall[RK_k] = RecallAggregator()

			order = np.argsort(-scores_K)[0:RK]
			cur_matches = matches_K[order]
			cur_relids = relids_K[order]
			recall_RK_k.update(np.sum(cur_matches), GT)

			if mean_recall is None:
				continue

			mean_recall_RK_k = mean_recall.get(RK_k, None)
			if not mean_recall_RK_k:
				mean_recall_RK_k = mean_recall[RK_k] = {}

			for relid,cnt in relcnt.items():
				mean_recall_RK_k_rel = mean_recall_RK_k.get(relid, None)
				if not mean_recall_RK_k_rel:
					mean_recall_RK_k_rel = mean_recall_RK_k[relid] = RecallAggregator()
				mean_recall_RK_k_rel.update(np.dot(cur_matches, cur_relids == relid), cnt)

def calc_mean_recall(relmap):
	rellist_local = []
	rellist_global = []
	for agg in relmap.values():
		r_local,r_global = agg.result()
		rellist_local.append(r_local)
		rellist_global.append(r_global)
	return sum(rellist_local) / tn_data.NUM_RELS, sum(rellist_global) / tn_data.NUM_RELS

recall = {}
mean_recall = {}
zs_recall = {}
seen_triplets = set()

for img in tqdm(trainimgs):
	for rel in img['rels']:
		src = rel['sv']
		dst = rel['dv']
		for relid in rel['v']:
			seen_triplets.add((src,dst,relid))

numimgs = 0
for img in tqdm(testimgs):
	id = img['id']
	with stupid_adapter(zf_scores.open(f'{id}.npy','r')) as f:
		all_scores = np.load(f)

	pairs = generate_pairs_for_preddet(all_scores, len(img['objs']))
	if pairs is None:
		print(f'Image with problem: {id}')
		continue

	# Preprocess ground truth
	ground_truth = set()
	zs_ground_truth = set()
	annotated_pairs = set()
	for rel in img['rels']:
		src = rel['si']
		dst = rel['di']
		srcv = rel['sv']
		dstv = rel['dv']
		annotated_pairs.add((src,dst))
		for relid in rel['v']:
			triplet = (src,dst,relid)
			ground_truth.add(triplet)
			if (srcv,dstv,relid) not in seen_triplets:
				zs_ground_truth.add(triplet)

	if len(ground_truth) > 0:
		matches, scores, relids = extract_scores(ground_truth, annotated_pairs, pairs)
		update_recall(recall, mean_recall, ground_truth, matches, scores, relids)

	if len(zs_ground_truth) > 0:
		matches, scores, relids = extract_scores(zs_ground_truth, annotated_pairs, pairs)
		update_recall(zs_recall, None, zs_ground_truth, matches, scores, relids)

for RK_k,agg in recall.items():
	recall[RK_k] = agg.result()

for RK_k,agg in zs_recall.items():
	zs_recall[RK_k] = agg.result()

for RK_k,relmap in mean_recall.items():
	mean_recall[RK_k] = calc_mean_recall(relmap)

print()

with open(f'test-results/{MODEL_VARIANT}.log', 'w', encoding='utf-8') as fout:
	def print_both(text):
		print(text)
		print(text,file=fout)

	def sort_key(key):
		RK,k = key[0]
		return (k,RK)

	def print_metric(name,metrics):
		for (RK,k),(v_local,v_global) in sorted(metrics.items(), key=sort_key):
			print_both(f'|{name:>7}@{RK:<3} k={k:<3} | {100*v_local:4.1f}% | {100*v_global:4.1f}% |')

	print_both(f'| ~~~~ Metric ~~~~ | Value | ~Alt~ |')
	print_both(f'------------------------------------')
	print_metric('R', recall)
	print_metric('zR', zs_recall)
	print_metric('mR', mean_recall)
