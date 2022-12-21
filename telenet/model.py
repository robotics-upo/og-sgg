import tensorflow as tf
import numpy as np

def Mask2Vec(N):
	return tf.keras.Sequential([
		tf.keras.layers.Input(shape=(64,64,2)),
		tf.keras.layers.Conv2D(96, kernel_size=5, strides=2, padding='same', kernel_initializer='he_uniform'),
		tf.keras.layers.ReLU(),
		tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same', kernel_initializer='he_uniform'),
		tf.keras.layers.ReLU(),
		tf.keras.layers.Conv2D(64, kernel_size=8, strides=1, padding='valid', kernel_initializer='he_uniform'),
		tf.keras.layers.ReLU(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(N, activation=tf.nn.relu, kernel_initializer='he_uniform')
	])

class NonVisualVectorGenerator(tf.keras.Model):
	def __init__(self, N, embed_size=250, **kwargs):
		super().__init__(**kwargs)
		self.mask2vec = Mask2Vec(N)
		self.d1 = tf.keras.layers.Dense(N,input_shape=(embed_size*2,),activation=tf.nn.relu,kernel_initializer='he_uniform')
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.d2 = tf.keras.layers.Dense(N,activation=tf.nn.relu,kernel_initializer='he_uniform')
		self.bn2 = tf.keras.layers.BatchNormalization()

	@tf.function
	def call(self, input, training=False):
		obj_mask, sem_vec1, sem_vec2 = input
		mask_vec = self.mask2vec(obj_mask, training=training)
		sem_vec = self.d1(tf.concat([sem_vec1, sem_vec2], axis=-1), training=training)
		sem_vec = self.bn1(sem_vec, training=training)
		sem_vec = self.d2(sem_vec, training=training)
		sem_vec = self.bn2(sem_vec, training=training)
		return tf.concat([mask_vec, sem_vec], axis=-1)

class RelationshipDetector(tf.keras.Model):
	def __init__(self, N, base_len, num_iters=5, **kwargs):
		super().__init__(**kwargs)
		self.g0 = tf.keras.layers.GlobalAvgPool2D()
		self.d1 = tf.keras.layers.Dense(base_len*2, input_shape=(base_len*3,), activation=tf.nn.relu, kernel_initializer='he_uniform')
		self.bn1 = tf.keras.layers.BatchNormalization()
		self.d2 = tf.keras.layers.Dense(base_len, activation=tf.nn.relu, kernel_initializer='he_uniform')
		self.bn2 = tf.keras.layers.BatchNormalization()
		self.gru = tf.keras.layers.GRU(base_len)
		self.attention = tf.keras.Sequential([
			tf.keras.layers.Reshape((1,1,-1), input_shape=(base_len,)),
			tf.keras.layers.Conv2DTranspose(base_len//4, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_uniform'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Conv2DTranspose(base_len//16, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_uniform'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Conv2DTranspose(base_len//64, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_uniform'),
			tf.keras.layers.ReLU(),
			tf.keras.layers.Conv2DTranspose(1, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_uniform'),
			tf.keras.layers.Resizing(10, 10),
			tf.keras.layers.Softmax(axis=[1,2])
		])
		self.numiters = num_iters
		self.scoring = tf.keras.layers.Dense(N, input_shape=(base_len*3,), dtype='float32', kernel_initializer='orthogonal')

	@tf.function
	def call(self, input, training=False):
		nonvis_vec, featuremap, prior = input

		vis_vec = self.g0(featuremap, training=training)
		if nonvis_vec.shape[0] == vis_vec.shape[0]:
			combined_vec = tf.concat([nonvis_vec, vis_vec], axis=-1)
		elif vis_vec.shape[0] == 1:
			vis_vec = tf.squeeze(vis_vec, axis=0)
			combined_vec = tf.vectorized_map(lambda x: tf.concat([x, vis_vec], axis=-1), elems=nonvis_vec)
		else:
			raise Exception("Invalid shapes for visual/non-visual vectors")
		score = self.scoring(combined_vec, training=training)

		gru_state = None
		for i in range(self.numiters):
			gru_input = self.d1(combined_vec, training=training)
			gru_input = self.bn1(gru_input, training=training)
			gru_input = self.d2(gru_input, training=training)
			gru_input = self.bn2(gru_input, training=training)
			gru_input = tf.expand_dims(gru_input, axis=1)
			gru_state = self.gru(gru_input, initial_state=gru_state, training=training)

			attnmask = self.attention(gru_state, training=training)
			vis_vec = tf.reduce_sum(featuremap * attnmask, axis=[1,2], keepdims=False)

			combined_vec = tf.concat([nonvis_vec, vis_vec], axis=-1)
			score += self.scoring(combined_vec, training=training)

		score /= self.numiters + 1
		score += prior
		return score

class CombinedRelationshipDetector(tf.keras.Model):
	def __init__(self, N, sem_embed_size=250, **kwargs):
		super().__init__(**kwargs)
		self.vecgen = NonVisualVectorGenerator(512, embed_size=sem_embed_size)
		self.reldec = RelationshipDetector(N, base_len=512)

	@tf.function
	def call(self, input, training=False):
		feature_map, obj_mask, sem_vec1, sem_vec2, prior = input
		if len(feature_map.shape) == 3:
			feature_map = tf.expand_dims(feature_map, axis=0)
		if obj_mask.shape[1] == 2:
			obj_mask = tf.transpose(obj_mask, perm=(0,2,3,1))
		nonvis_vec = self.vecgen((obj_mask, sem_vec1, sem_vec2), training=training)
		x= self.reldec((nonvis_vec, feature_map, prior), training=training)
		return x

class TrivialRelationshipDetector(tf.keras.Model):
	def __init__(self, N, sem_embed_size=250, **kwargs):
		super().__init__(**kwargs)
		self.dense = tf.keras.layers.Dense(50*16, input_shape=(sem_embed_size*2,), activation=tf.nn.relu,kernel_initializer='he_uniform')
		self.bn = tf.keras.layers.BatchNormalization()
		self.scoring = tf.keras.layers.Dense(N, dtype='float32', kernel_initializer='orthogonal')

	@tf.function
	def call(self, input, training=False):
		feature_map, obj_mask, sem_vec1, sem_vec2, prior = input
		x = tf.concat([sem_vec1, sem_vec2], axis=-1)
		x = self.dense(x, training=training)
		x = self.bn(x, training=training)
		x = self.scoring(x, training=training)
		return x

class SlightlyLessTrivialRelationshipDetector(tf.keras.Model):
	def __init__(self, N, sem_embed_size=250, **kwargs):
		super().__init__(**kwargs)
		self.mask2vec = Mask2Vec(sem_embed_size)
		self.dense = tf.keras.layers.Dense(50*24, input_shape=(sem_embed_size*3,), activation=tf.nn.relu,kernel_initializer='he_uniform')
		self.bn = tf.keras.layers.BatchNormalization()
		self.scoring = tf.keras.layers.Dense(N, dtype='float32', kernel_initializer='orthogonal')

	@tf.function
	def call(self, input, training=False):
		feature_map, obj_mask, sem_vec1, sem_vec2, prior = input
		obj_vec = self.mask2vec(obj_mask)
		x = tf.concat([sem_vec1, sem_vec2, obj_vec], axis=-1)
		x = self.dense(x, training=training)
		x = self.bn(x, training=training)
		x = self.scoring(x, training=training)
		return x
