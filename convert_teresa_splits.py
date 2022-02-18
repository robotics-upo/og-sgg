import telenet.dataset_data as tn_data
from telenet.config import get as tn_config
import re
import os
import json
import lzma

TERESA_DIR = tn_config('paths.teresa')

with open(f'{TERESA_DIR}/handpicked-labels/labels.txt', 'rt', encoding='utf-8') as f:
	TERESA_NAMES = f.read().rstrip('\n').split('\n')

SUBPATT_ID = "([0-9a-zA-Z_]+)#(\d+)"
SUBPATT_FILE = "[0-9a-fA-F]+"
PATT_NEWFILE = re.compile(f"^{SUBPATT_FILE}$")
PATT_NEWFILE2 = re.compile(f"^({SUBPATT_FILE}).[tT][xX][tT]$")
PATT_TRIPLET = re.compile(f"^{SUBPATT_ID}\s+(.+?)\s+{SUBPATT_ID}$")

images = {}
fancymaps = {}
for fil in os.scandir(f'{TERESA_DIR}/handpicked-labels'):
	if not fil.is_file() or not (m := PATT_NEWFILE2.match(fil.name)):
		continue
	nam = m[1]
	img = images[nam] = { 'id': nam, 'w': 640, 'h': 480, 'objs': [], 'rels': [] }
	fm = fancymaps[nam] = {}
	curcnt = {}
	with open(fil.path, 'r', encoding='utf-8') as f:
		while (line := f.readline().strip()):
			m = line.split(' ')
			assert len(m) == 5
			id,xc,yc,w,h = m
			x = float(xc)-.5*float(w); y = float(yc)-.5*float(h)
			x = int(.5+640.*x); y = int(.5+480.*y)
			w = int(.5+640.*float(w)); h = int(.5+480.*float(h))
			id = int(id)
			curid = curcnt[id] = 1+curcnt.get(id,0)
			fm[f"{TERESA_NAMES[id].replace(' ','')}#{curid}"] = len(img['objs'])
			img['objs'].append({ 'v': id, 'bb': [x,y,w,h]})

tripletdb = {}
relnames = []
relnametoid = {}
with open(f'{TERESA_DIR}/handpicked-labels/relations.txt', 'r', encoding='utf-8') as f:
	while (line := f.readline().strip()):
		if PATT_NEWFILE.match(line):
			fm = fancymaps[line]
			tr = tripletdb[line] = {}
		elif m := PATT_TRIPLET.match(line):
			srccls = m[1]; srcidx = int(m[2])
			dstcls = m[4]; dstidx = int(m[5])
			rel = m[3]
			srcdst = (fm[f'{srccls}#{srcidx}'], fm[f'{dstcls}#{dstidx}'])
			rels = tr.get(srcdst, None)
			if not rels: rels = tr[srcdst] = set()
			relid = relnametoid.get(rel,-1)
			if relid < 0:
				relid = relnametoid[rel] = len(relnames)
				relnames.append(rel)
			rels.add(relid)

for id,tr in tripletdb.items():
	img = images[id]
	for (src,dst),rels in tr.items():
		img['rels'].append({
			'si': src,
			'sv': img['objs'][src]['v'],
			'di': dst,
			'dv': img['objs'][dst]['v'],
			'v': list(rels)
		})

with open('testdata/teresa-names.json', 'wt', encoding='utf-8') as f:
	json.dump({ 'objs': TERESA_NAMES, 'attrs': [], 'rels': relnames }, f)

with lzma.open(f'testdata/teresa-test.json.xz', 'wt', encoding='utf-8') as f:
	json.dump(list(images.values()), f)
