import io
import zipfile
import os
import random

from telenet.config import get as tn_config

RND_SEED = tn_config('train.random_seed')

os.environ['PYTHONHASHSEED'] = str(RND_SEED)
random.seed(RND_SEED)

import numpy as np
import pandas as pd

np.random.seed(RND_SEED)

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_docs as tfdocs
import tensorflow_docs.plots # Do not remove this import
#import tensorflow.keras.mixed_precision as mp

from tensorflow.python.training.tracking.data_structures import NoDependency

tf.random.set_seed(RND_SEED)

from matplotlib import pyplot as plt

import telenet.model as tn_model
import telenet.dataset_data as tn_data

from tqdm import tqdm

#mp.set_global_policy(mp.Policy('mixed_float16'))

DATASET_NAME = tn_config('train.dataset')
MODEL_VARIANT = tn_config('model.variant')

if 'teresa' in DATASET_NAME:
	tn_data.load_names(f'teresa-names.json')
else:
	tn_data.load_names(f'{DATASET_NAME}-names.json')

TRAIN_DATA = tn_data.load_json_xz(f'{DATASET_NAME}-train-without-val')
VAL_DATA = tn_data.load_json_xz(f'{DATASET_NAME}-val')

for known_dataset in ['vrd', 'vg', DATASET_NAME]:
	if DATASET_NAME.startswith(known_dataset):
		SEM_VECTORS = tf.convert_to_tensor(np.load(tn_data.path(f'{known_dataset}-semvecs.npy')))
		zf_pi = zipfile.ZipFile(tn_data.path(f'{known_dataset}-yolo-train.zip'), 'r')
		zf_om = zipfile.ZipFile(tn_data.path(f'{known_dataset}-mask-train.zip'), 'r')

def get_priors(src,dst):
	return np.zeros((tn_data.NUM_RELS,), np.float32)

def preprocess_gt_nongt(img):
	img['gt'] = gt = {}
	for rel in img['rels']:
		src_id = rel['si']
		dst_id = rel['di']
		y_real = np.zeros((tn_data.NUM_RELS,), np.float32)
		for i in rel['v']:
			y_real[i] = 1.
		gt[(src_id,dst_id)] = {
			'sv': rel['sv'],
			'dv': rel['dv'],
			'a': SEM_VECTORS[rel['sv']],
			'b': SEM_VECTORS[rel['dv']],
			'p': get_priors(rel['sv'],rel['dv']),
			'y': y_real
		}
	img['nongt'] = nongt = set()
	for i in range(len(img['objs'])):
		for j in range(len(img['objs'])):
			if i != j and (i,j) not in gt:
				nongt.add((i,j))

# Preprocess training/validation data
for img in TRAIN_DATA:
	preprocess_gt_nongt(img)
for img in VAL_DATA:
	preprocess_gt_nongt(img)

def stupid_adapter(f):
	return io.BytesIO(f.read())

class TelenetTrainer(tn_model.CombinedRelationshipDetector):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.trn_batch_size = NoDependency(32)
		self.trn_batch_gt_size = NoDependency(self.trn_batch_size - int(.5 + .5 * self.trn_batch_size))
		self.trn_loss_tracker = NoDependency(tf.keras.metrics.Mean(name="loss"))
		self.val_loss_tracker = NoDependency(tf.keras.metrics.Mean(name="loss"))

	def prepare_minibatch(self, img):
		img_name = img['id']
		with stupid_adapter(zf_pi.open(f'{img_name}.npy','r')) as f:
			img_features = tf.expand_dims(tf.convert_to_tensor(np.load(f), tf.float32), axis=0)
		with stupid_adapter(zf_om.open(f'{img_name}.npy','r')) as f:
			obj_masks = tf.convert_to_tensor(np.load(f)[0,:,:,:], tf.float32)

		num_objs = len(img['objs'])
		num_pairs = num_objs * (num_objs - 1)
		if num_pairs == 0:
			return (None, None, None)

		ground_truth = img['gt']
		non_ground_truth = img['nongt']
		num_gt_pairs = len(ground_truth)
		num_non_gt_pairs = len(non_ground_truth)
		batch_mask = []
		batch_srcsem = []
		batch_dstsem = []
		batch_priors = []
		batch_y_real = []

		def sample_gt_pair(pair, pairdata):
			src_id,dst_id = pair
			batch_mask.append(tf.stack([obj_masks[:,:,src_id], obj_masks[:,:,dst_id]], axis=-1))
			batch_srcsem.append(pairdata['a'])
			batch_dstsem.append(pairdata['b'])
			batch_priors.append(pairdata['p'])
			batch_y_real.append(pairdata['y'])

		def sample_non_gt_pair(pair):
			src_id,dst_id = pair
			src_objid = img['objs'][src_id]['v']
			dst_objid = img['objs'][dst_id]['v']
			batch_mask.append(tf.stack([obj_masks[:,:,src_id], obj_masks[:,:,dst_id]], axis=-1))
			batch_srcsem.append(SEM_VECTORS[src_objid])
			batch_dstsem.append(SEM_VECTORS[dst_objid])
			batch_priors.append(get_priors(src_objid, dst_objid))
			batch_y_real.append(np.zeros((tn_data.NUM_RELS,), np.float32))

		num_sampled_gt_pairs = np.minimum(self.trn_batch_gt_size, num_gt_pairs)
		num_sampled_non_gt_pairs = np.minimum(self.trn_batch_size - num_sampled_gt_pairs, num_non_gt_pairs)
		num_dupes = self.trn_batch_size - num_sampled_gt_pairs - num_sampled_non_gt_pairs

		for pair,pairdata in random.sample(list(ground_truth.items()), k=num_sampled_gt_pairs):
			sample_gt_pair(pair, pairdata)
		for pair in random.sample(list(non_ground_truth), k=num_sampled_non_gt_pairs):
			sample_non_gt_pair(pair)

		# Fill batch with dupes
		if num_dupes > 0:
			for i in random.choices(list(range(len(batch_mask))), k=num_dupes):
				batch_mask.append(batch_mask[i])
				batch_srcsem.append(batch_srcsem[i])
				batch_dstsem.append(batch_dstsem[i])
				batch_priors.append(batch_priors[i])
				batch_y_real.append(batch_y_real[i])

		batch_mask = tf.stack(batch_mask, axis=0)
		batch_srcsem = tf.stack(batch_srcsem, axis=0)
		batch_dstsem = tf.stack(batch_dstsem, axis=0)
		batch_priors = tf.stack(batch_priors, axis=0)
		batch_y_real = tf.stack(batch_y_real, axis=0)
		batch_x = (img_features, batch_mask, batch_srcsem, batch_dstsem, batch_priors)
		return (batch_x, batch_y_real)

	@property
	def metrics(self):
		return [self.trn_loss_tracker, self.val_loss_tracker]

	def ranking_loss(self, y_real, y_pred, margin=1.):
		scores_0, scores_1 = tf.dynamic_partition(y_pred, tf.cast(y_real, tf.int32), 2)
		scale = tf.size(y_real, out_type=tf.float32)
		return tf.reduce_sum(tf.vectorized_map(lambda val: tf.reduce_sum(tf.nn.relu(margin - (scores_1 - val))), elems=scores_0)) / scale

	@tf.function
	def train_kernel(self, x, y_real):
		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			loss = self.ranking_loss(y_real, y_pred)

		self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
		return loss

	def train_step(self, data):
		batch_x, batch_y_real = self.prepare_minibatch(TRAIN_DATA[int(data)])
		if batch_x is not None:
			loss = self.train_kernel(batch_x, batch_y_real)
			self.trn_loss_tracker.update_state(loss)
		return { 'loss': self.trn_loss_tracker.result() }

	@tf.function
	def test_kernel(self, x, y_real):
		y_pred = self(x, training=False)
		return self.ranking_loss(y_real, y_pred)

	def test_step(self, data):
		batch_x, batch_y_real = self.prepare_minibatch(VAL_DATA[int(data)])
		if batch_x is not None:
			loss = self.test_kernel(batch_x, batch_y_real)
			self.val_loss_tracker.update_state(loss)
		return { 'loss': self.val_loss_tracker.result() }

mdl = TelenetTrainer(N=tn_data.NUM_RELS)
mdl.compile(
	optimizer=tfa.optimizers.AdamW(learning_rate=tn_config('train.lr'), weight_decay=tn_config('train.wd')),
	run_eagerly=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=tn_config('train.early_stopping'),
	mode='min',
	restore_best_weights=True
)

tensorboard = tf.keras.callbacks.TensorBoard(
	log_dir=f'tensorboard/{MODEL_VARIANT}',
	histogram_freq=1
)

history = mdl.fit(
	x = tf.data.Dataset.range(len(TRAIN_DATA)).shuffle(256, seed=RND_SEED, reshuffle_each_iteration=True),
	validation_data = tf.data.Dataset.range(len(VAL_DATA)),
	callbacks = [ early_stopping, tensorboard ],
	epochs = tn_config('train.epochs')
)

mdl.save_weights(f'weights/telenet+{MODEL_VARIANT}')

plt.figure()
plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
plotter.plot({ 'Model': history })
plt.savefig(f"train-results/{MODEL_VARIANT}.png")

with open(f'train-results/{MODEL_VARIANT}.csv', mode='wt', encoding='utf-8') as f:
	pd.DataFrame(history.history).to_csv(f)
