import tensorflow as tf
import numpy as np

from .config import get as tn_config

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		#print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

mdl_yolo = tf.saved_model.load(tn_config('paths.yolo'))
mdl_yolo.trainable = False

def load_image_for_vrd_yolo(data):
	img = tf.cast(tf.image.decode_jpeg(data, channels=3), tf.float32)
	h,w,_ = img.shape
	img = tf.image.resize(img, (320, 320)) / 255.
	return img, w, h

def load_image_from_data(data, preprocess=True):
	img = tf.cast(tf.image.decode_jpeg(data, channels=3), tf.float32)
	img = tf.image.resize(img, (320, 320)) / 255.
	if preprocess:
		img = tf.expand_dims(img, 0)
		img,_ = mdl_yolo(img)
	return img

def load_image(path):
	return load_image_from_data(tf.io.read_file(path))

def resized_range(oldsz, newsz):
	if oldsz == newsz:
		for i in range(newsz):
			yield (i,slice(i,i+1,1))
	elif oldsz < newsz:
		for i in range(newsz):
			j = int(i*oldsz/newsz)
			yield (i,slice(j,j+1,1))
	else: # oldsz > newsz
		nextj = 0
		for i in range(newsz):
			j = nextj
			nextj = int(0.5+(i+1)*oldsz/newsz)
			yield (i,slice(j,nextj,1))

def safe_resize(img, neww, newh):
	out = np.zeros((newh, neww, img.shape[2]))
	for y,oldy in resized_range(img.shape[0], newh):
		for x,oldx in resized_range(img.shape[1], neww):
			out[y,x,:] = np.max(img[oldy,oldx,:],axis=(0,1))
	return out

def bbox_masks(width, height, boxes):
	mask = np.zeros((height, width, len(boxes)))
	for i in range(len(boxes)):
		bb = boxes[i] # 0=xmin 1=xmax 2=ymin 3=ymax
		mask[bb[2]:bb[3],bb[0]:bb[1],i] = 1.0
	return mask

def create_bbox_mask(width, height, *args):
	mask = bbox_masks(width, height, args)
	mask = safe_resize(mask, 64, 64)
	mask = tf.convert_to_tensor(mask)
	mask = tf.expand_dims(mask, 0)
	#mask = tf.image.resize(tf.expand_dims(mask, 0), (64,64), method=tf.image.ResizeMethod.GAUSSIAN, antialias=True)
	return mask

def create_bbox_depth_mask(depthimg, *args):
	mask = depthimg * bbox_masks(depthimg.shape[1], depthimg.shape[0], args)
	mask = safe_resize(mask, 64, 64)
	mask = tf.convert_to_tensor(mask)
	mask = tf.expand_dims(mask, 0)
	return mask

def decode_yolo_bbox(bb,w,h):
	ymin,xmin,ymax,xmax = bb
	return int(.5+xmin*w), int(.5+xmax*w), int(.5+ymin*h), int(.5+ymax*h)

def parse_yolo_results(results, width, height, iou_thr=0.5, score_thr=0.4):
	boxes = results[:,:,0:4]
	scores = results[:,:,4:]
	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
		scores=tf.reshape(
			scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
		max_output_size_per_class=50,
		max_total_size=50,
		iou_threshold=iou_thr,
		score_threshold=score_thr
	)
	num_objs = int(valid_detections[0])
	objs = []
	cnt = {}
	for i in range(num_objs):
		box = decode_yolo_bbox(boxes[0,i,:],width,height)
		score = float(scores[0,i])
		clsid = int(classes[0,i])
		cnt[clsid] = cnt.get(clsid, 0) + 1
		objs.append((clsid,cnt[clsid],score,box))
	return objs
