# CheckedBox

Simple NN to classify check-box in three classes: empty, right and wrong.

## Requirements

- Task (optional but highly suggested)
- Python## Requirements

## Setup

1. create the virtual envirionment (_venv_): `task venv`
2. activate the _venv_: `source .venv/bin/activate`

## Usage

When the _venv_ is activated the command **run_dnn** is available:

```bash
usage: run_dnn [-h] [-d DATASET] [-e EPOCHS] [-b BATCHES] [-r RATE] [-m MODEL] [-g] [-s {pth,onnx,not}]

options:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        path to the dataset (default: /mnt/archive/work/NiNGia/checkedbox/data)
  -e EPOCHS, --epochs EPOCHS
                        number of epoch for the training (default: 5)
  -b BATCHES, --batches BATCHES
                        number of batches (default: 64)
  -r RATE, --rate RATE  learning rate (default: 0.001)
  -m MODEL, --model MODEL
                        model to use (default: NGConvNet)
  -g, --in-gray
  -s {pth,onnx,not}, --save {pth,onnx,not}
```

<!-- ## TOWRITE
* how to train a new model
* how to update the dataset -->

<!-- run_dnn --epochs 100 -d /home/sergio/Documenti/used_dataset/dataset -->
