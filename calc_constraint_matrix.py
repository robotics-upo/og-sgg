import lzma

import numpy as np
import owlready2 as owl

import telenet.dataset_data as tn_data
from telenet.config import get as tn_config

from tqdm import tqdm
from pathlib import Path

DATASET_NAME = 'ai2thor'
tn_data.load_names(f'{DATASET_NAME}-names.json')
owl.JAVA_EXE = tn_config('paths.java')
onto = owl.get_ontology(Path(tn_data.path(f'{DATASET_NAME}.owl')).as_uri()).load()
with onto: owl.sync_reasoner()

ONTO_CLS  = [ onto.search_one(label=label.replace(' ','')) for label in tn_data.CLASS_NAMES ]
ONTO_RELS = [ onto.search_one(label=label) for label in tn_data.REL_NAMES ]

def is_compat(lhs, rhs):
	if lhs == rhs:
		return True
	if owl.issubclass(rhs, owl.Thing):
		return owl.issubclass(lhs, rhs)
	rhs_ = type(rhs)
	if not owl.issubclass(rhs_, owl.ClassConstruct):
		raise Exception("Dunno what to do with " + str(rhs))
	if rhs_ is owl.Not:
		return not is_compat(lhs, rhs.Class)
	if rhs_ is owl.And:
		return all(is_compat(lhs, c) for c in rhs.Classes)
	if rhs_ is owl.Or:
		return any(is_compat(lhs, c) for c in rhs.Classes)
	raise Exception("Unsupported construct " + str(rhs))

def calc_compat(intlist):
	acc = np.ones((tn_data.NUM_CLASSES,), np.bool8)
	for q in intlist:
		acc = np.logical_and(acc, np.array([ is_compat(clz, q) for clz in ONTO_CLS ], np.bool8))
	return acc

relcompat = []
for rel in ONTO_RELS:
	print(rel.label[0])
	print('  Transitive:   ', issubclass(rel, owl.TransitiveProperty))
	print('  Functional:   ', issubclass(rel, owl.FunctionalProperty))
	print('  InvFunctional:', issubclass(rel, owl.InverseFunctionalProperty))
	print('  Domain:', rel.domain)
	print('  Range:', rel.range)
	domain = calc_compat(rel.domain)
	#print(domain)
	range  = calc_compat(rel.range)
	#print(range)
	relcompat.append(np.outer(domain, range))

relcompat = np.stack(relcompat, axis=-1)

with lzma.open(tn_data.path(f'{DATASET_NAME}-pair-constraints.npy.xz'), 'wb') as f:
	np.save(f, relcompat)
