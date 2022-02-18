# OG-SGG

This repository contains the code developed for the paper
"OG-SGG: Ontology-Guided Scene Graph Generation. A Case Study in Transfer Learning for Telepresence Robotics".

## Environment

A suitable environment can be prepared using the Conda/Mamba package manager:

```bash
conda create -n tf cudatoolkit=11.3 cudnn=8.2 python=3.9
conda activate tf
conda install cuda-nvcc=11.3 -c nvidia
mamba install pandas matplotlib py-opencv python-graphviz tqdm pytomlpp -c conda-forge
pip install tensorflow tensorflow-addons tensorflow-hub
git clone https://github.com/tensorflow/docs tfdocs
cd tfdocs && python setup.py install && cd ..
```

The environment can be verified to work using the following python commands:

```bash
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

## Datasets

### VG

Download the following files belonging to the [Visual Genome dataset](https://visualgenome.org/api/v0/api_home.html):

- [images.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) (*do not extract zip*)
- [images2.zip](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip) (*do not extract zip*)
- [image_data.json](https://visualgenome.org/static/data/dataset/image_data.json.zip) (*extract json from zip*)
- [VG-SGG.h5](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5)
- [VG-SGG-dicts.json](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json)

```
vg/
|___ images.zip
|___ images2.zip
|___ image_data.json
|___ VG-SGG.h5
|___ VG-SGG-dicts.json
```

### TERESA

[Download TERESA test set](https://robotics.upo.es/~famozur/ogsgg/ogsgg-teresa.zip):

```
teresa/
|___ <image.jpg> ...
|___ <image.txt> ...
|___ labels.txt
|___ relations.txt
```

## Network

This repository contains an implementation of [VRD-RANS](https://www.sciencedirect.com/science/article/pii/S0925231220320117), the scene graph generation network we chose to base our research upon.

```tex
@article{WANG202155,
	title = {{Visual relationship detection with recurrent attention and negative sampling}},
	journal = {Neurocomputing},
	volume = {434},
	pages = {55-66},
	year = {2021},
	issn = {0925-2312},
	doi = {https://doi.org/10.1016/j.neucom.2020.12.099},
	url = {https://www.sciencedirect.com/science/article/pii/S0925231220320117},
	author = {Lei Wang and Peizhen Lin and Jun Cheng and Feng Liu and Xiaoliang Ma and Jianqin Yin},
	keywords = {Computer vision, Neural networks, Visual relations},
}
```

## Pre-trained weights

[Download pre-trained weights used in the paper](https://robotics.upo.es/~famozur/ogsgg/ogsgg-weights.zip), and extract all files to a new `weights/` folder:

VG-SGG models:

- `VgSgg+NoFilter`: Baseline
- `VgSgg+Filter+NoAug`: TERESA filter, no augmentation
- `VgSgg+Filter+WithAug`: TERESA filter, with augmentation

VG-indoor models:

- `VgIndoor+NoFilter`: Baseline
- `VgIndoor+Filter+NoAug`: TERESA filter, no augmentation
- `VgIndoor+Filter+WithAug`: TERESA filter, with augmentation

## Configuration

The codebase can be configured using `config.toml`. There are several sections:

- `[paths]`: Contains paths to datasets and other resources.
- `[model]`: Contains global model configuration.
- `[train]`: Configures the training stage.
- `[test]`: Configures the testing inference stage.
- `[convert]`: Configures the dataset filter/augmentation stage.
- `[stratify]`: Configures the dataset stratification stage.
- `[teresa.qualitative]`: Configures rules for qualitative results.
- `[teresa.predicate_map]`: Configures the ontology predicate map.

## Scripts

Loading VG:

- `convert_vg_splits.py`: Loads training/test splits from VG-SGG and converts them to the format used in this codebase.
- `convert_vg_masks.py`: Precalculates VG object masks.
- `convert_vg_images.py`: Preconverts VG images into feature maps using the YOLO backbone.
- `stratify.py`: Extracts an evaluation set from the training split, respecting predicate distribution (i.e. through stratification).

Loading TERESA:

- `convert_teresa_splits.py`: Loads TERESA's test split and converts it to the format used in this codebase.
- `convert_teresa_images.py`: Precalculates TERESA object masks.
- `convert_teresa_masks.py`: Precalculates TERESA feature maps.
- `calc_teresa_constraint_matrix.py`: Precalculates TERESA domain/range constraint matrix.

Filtering and preprocessing:

- `new_filter.py`: Generates VG-indoor from VG-SGG.
- `conv_vg_to_teresa.py`: Generates an adapted, filtered (and augmented) version of VG-SGG or VG-indoor using TERESA ontology.
- `dataset_stats.py`: Prints dataset statistics.

Training and testing:

- `train_telenet.py`: Trains VRD-RANS on the specified dataset.
- `test_telenet.py`: Generates testing data for the model using the specified test set.
- `calc_metrics_for_teresa.py`: Calculates R@K family of metrics on the TERESA test set, including the ontology-guide post-processing logic.
- `teresa_qualitative.py`: Generates qualitative results on the TERESA test set, including the ontology-guide post-processing logic.

## Reference

```tex
@article{ogsgg2022,
	title={{OG-SGG: Ontology-Guided Scene Graph Generation. A Case Study in Transfer Learning for Telepresence Robotics}},
	author={Amodeo, Fernando and Caballero, Fernando and Díaz-Rodriguez, Natalia and Merino, Luis},
	year={2022},
}
```
