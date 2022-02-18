import os
import pytomlpp

CONFIG_FILE = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../config.toml')
CONFIG = pytomlpp.load(CONFIG_FILE)

def get(key:str):
	pos = CONFIG
	for t in key.split('.'):
		pos = pos.get(t, None)
		if pos is None:
			raise Exception(f'Unknown config key: {key}')
	return pos
