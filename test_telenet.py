import zipfile
import io

import tensorflow as tf
import numpy as np

import telenet.model as tn_model
import telenet.dataset_data as tn_data
from telenet.config import get as tn_config

from tqdm import tqdm

DATASET_NAME = tn_config('train.dataset')
MODEL_VARIANT = tn_config('model.variant')

TARGET_DATASET = tn_config('test.dataset')

if TARGET_DATASET in DATASET_NAME: DATASET_NAME = TARGET_DATASET

tn_data.load_names(f'{DATASET_NAME}-names.json')
SEM_VECTORS = np.load(tn_data.path(f'{TARGET_DATASET}-semvecs.npy'))
zf_yolo = zipfile.ZipFile(tn_data.path(f'{TARGET_DATASET}-yolo-test.zip'), 'r')
zf_mask = zipfile.ZipFile(tn_data.path(f'{TARGET_DATASET}-mask-test.zip'), 'r')
testimgs = tn_data.load_json_xz(f'{TARGET_DATASET}-test')

def decode_bbox(bb,sz):
	w,h = sz
	ymin,xmin,ymax,xmax = bb
	return int(.5+xmin*w), int(.5+xmax*w), int(.5+ymin*h), int(.5+ymax*h)

def generate_image_relations(masks, objs):
	def generate():
		for i in range(len(objs)):
			for j in range(len(objs)):
				if i == j:
					continue
				obj_mask = np.stack([masks[:,:,i], masks[:,:,j]], axis=-1)
				src_obj = objs[i]['v']
				dst_obj = objs[j]['v']
				yield ((i,j), (obj_mask, SEM_VECTORS[src_obj], SEM_VECTORS[dst_obj], np.zeros((tn_data.NUM_RELS,), np.float32)))

	return tf.data.Dataset.from_generator(
		generate,
		output_signature=(
			tf.TensorSpec(shape=(2,), dtype=tf.int32),
			(
				tf.TensorSpec(shape=(64,64,2), dtype=tf.float32),
				tf.TensorSpec(shape=(250,), dtype=tf.float32),
				tf.TensorSpec(shape=(250,), dtype=tf.float32),
				tf.TensorSpec(shape=(tn_data.NUM_RELS,), dtype=tf.float32),
			),
		)
	)

def stupid_adapter(f):
	return io.BytesIO(f.read())

print('Loading TeleNet model...')

mdl = tn_model.CombinedRelationshipDetector(tn_data.NUM_RELS)
mdl.load_weights(f'weights/telenet+{MODEL_VARIANT}')

with zipfile.ZipFile(f'test-results/{TARGET_DATASET}+{MODEL_VARIANT}.zip', 'w') as zfo:
	for img in tqdm(testimgs):
		id = img['id']
		if len(img['objs']) < 2: continue

		with stupid_adapter(zf_yolo.open(f'{id}.npy','r')) as f:
			img_features = tf.expand_dims(np.load(f), axis=0)
		with stupid_adapter(zf_mask.open(f'{id}.npy','r')) as f:
			obj_masks = np.load(f)[0,:,:,:]

		scorelist = []
		for indices,batch in generate_image_relations(obj_masks, img['objs']).batch(16):
			batch_masks,batch_srcsem,batch_dstsem,batch_prior = batch
			batch = mdl((img_features,batch_masks,batch_srcsem,batch_dstsem,batch_prior))
			scorelist.append(batch.numpy())

		scores = np.concatenate(scorelist, axis=0)
		with zfo.open(f'{id}.npy','w') as f:
			np.save(f, scores)
