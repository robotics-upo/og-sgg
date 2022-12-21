# OG-SGG

This repository contains the code developed for the paper
["OG-SGG: Ontology-Guided Scene Graph Generation. A Case Study in Transfer Learning for Telepresence Robotics"](https://doi.org/10.1109/ACCESS.2022.3230590).

## Environment

A suitable environment can be prepared using the Conda/Mamba package manager ([mambaforge](https://github.com/conda-forge/miniforge#mambaforge) is recommended):

```bash
mamba create -n ogsgg
mamba activate ogsgg

mamba install -c conda-forge python=3.9 cudatoolkit=11.2 cudnn=8.1.0 owlready2=0.36 pandas matplotlib py-opencv python-graphviz tqdm pytomlpp
mamba clean -af

pip install tensorflow tensorflow-addons tensorflow-hub
pip install -U git+https://github.com/tensorflow/docs.git
pip cache purge

mamba env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
mamba env config vars set TF_CPP_MIN_LOG_LEVEL=1
mamba deactivate
mamba activate ogsgg
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

### AI2THOR

[Download AI2THOR test data](https://robotics.upo.es/~famozur/ogsgg/ai2thor-testdata.zip).
The zip archive should be decompressed, and the files within placed in the `testdata` folder:

```
testdata/
|___ ai2thor-images.zip
|___ ai2thor-test.json.xz
|___ ai2thor-names.json
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
- `new_filter.py`: Generates VG-indoor from VG-SGG.

Loading TERESA:

- `convert_teresa_splits.py`: Loads TERESA's test split and converts it to the format used in this codebase.
- `convert_teresa_images.py`: Precalculates TERESA feature maps.
- `convert_teresa_masks.py`: Precalculates TERESA object masks.

Loading AI2THOR:

- `convert_ai2thor_images.py`: Precalculates AI2THOR feature maps.
- `convert_ai2thor_masks.py`: Precalculates AI2THOR object masks.

OG-SGG helper scripts (filtering/augmentation, post-processing):

- `dataset_filter_aug.py`: Generates an adapted, filtered (and augmented) version of VG-SGG or VG-indoor using the specified ontology.
- `calc_constraint_matrix.py`: Precalculates domain/range constraint matrix.
- `dataset_stats.py`: Prints dataset statistics.

Training and testing:

- `train_telenet.py`: Trains VRD-RANS on the specified dataset.
- `test_telenet.py`: Generates testing data for the model using the specified test set.
- `ogsgg_quantitative.py`: Calculates R@K family of metrics on the given test set, including the ontology-guide post-processing logic.
- `ogsgg_qualitative.py`: Generates qualitative results on the given test set, including the ontology-guide post-processing logic.

## Reference

```tex
@article{ogsgg2022,
	journal={IEEE Access},
	title={{OG-SGG: Ontology-Guided Scene Graph Generation. A Case Study in Transfer Learning for Telepresence Robotics}},
	author={Amodeo, Fernando and Caballero, Fernando and Díaz-Rodríguez, Natalia and Merino, Luis},
	year={2022},
	volume={},
	number={},
	pages={1-1},
	doi={10.1109/ACCESS.2022.3230590}
}
```
