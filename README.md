# spectral-dropout
Deep Learning (Fall 2019) project: Spectral Dropout for Sim2Real transfer

### Steps to reproduce the results

- Clone the repository in your home folder
```
git clone --recurse-submodules https://github.com/gibernas/spectral-dropout.git
```
- Install the requirements from the `requirements.txt` file.
- Inside the directory `spectral-dropout`, download and extract the two datasets [real](https://drive.google.com/drive/folders/15YEDeMDU6bfS4UqapTyZCKLlzBHcg8qR) and [sim](https://drive.google.com/file/d/1MDGu2DE_SP-RrQqoA_8ntQsLQrdtNckm)
- Start the training with a command such as below. More details on arguments is mentioned later.
```
python3 train.py --workers 6 --gpu 6 --dataset sim,real --lr 0.001 --model VanillaCNN --epochs 200
```
- Evaluations on all datasets can be run by simply doing
```
python3 cross_evaluate.py
```
- To submit to a Duckietown Challenge, read [this](https://docs.duckietown.org/daffy/AIDO/out/index.html) link and look inside the `challenge-aido_LF-template-pytorch` folder.

### Code Organization:

- `train.py`: Python script to train the model. The following arguments are supported
```
Required:
    --host: Use "local" in general unless training on ETH clusters.
    --model: "VanillaCNN" or "SpectralDropoutCNN"
    --dataset: "real" or "sim"
Optional:
    --gpu: provide GPU number otherwise CPU is used 
    --epochs: Number of epochs (Default: 1)
    --batch_size: Batch size (Default: 16)
    --lr: Learning rate (Default: 1e-4)
    --validation_split: Percentage of data used for validation (Default: 0.2)
    --image_res: Resolution of image (not squared, just used for rescaling (Default: 64)
```

- `models.py`: Contains all model classes
- `losses.py`: Loss functions for training

- `cross_evaluate.py`: Python script to evaluate models on all "final" models. The "final" models are saved when training finishes, or is interrupted. They can be recognized by the keyword "final" in the name.

- `utils/`: contains utility functions