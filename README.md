# TODO
- Integrate [Higra](https://github.com/higra/Higra) to generate multiscale hierarchy
- Check training of orientation maps
- Check results with original caffe implementation

# Description
PyTorch implementation of [Convolutional Oriented Boundaries](https://github.com/kmaninis/COB)

## Differences w.r.t original implementation:
- ResNet50 from pyTorch model zoo, which differs from author's Caffe model (has batch normalization layers)
- Batch size of 16
- Base learning-rate is 1e-4 and is increased for "deeper" layers
- Weight initialization is gaussian/normal instead of constant

# Dependencies
Most of these are easily installed with your favorite package manager

- PyTorch >= 1.0
- Numpy
- Scipy
- imgaug
- opencv-python 
- tqdm
- imgaug
- sklearn
- tensorboardX
- higra

### With a docker
While you do not need to use a docker for this, this setup will allow you to run the project:\
Create the docker with:
```
docker run -it -v $(pwd)/:/workspace/ --name cobnet  --rm pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime bash
```
Install the dependencies:
```
apt update
apt install ffmpeg libsm6 libxext6 -y
pip install -r requirements.txt
```

# Dataset
Download and uncompress the following datasets/annotations:
- [Pascal VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) 
- [Pascal-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) 

This can be achieved using the following commands:
```
wget https://cs.stanford.edu/\~roozbeh/pascal-context/trainval.tar.gz -P data
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar -P data

tar -xvf data/VOCtrainval_11-May-2012.tar -C data
tar -xvf data/trainval.tar.gz -C data
```

# Training
On first run, the whole dataset (>10k images) will be processed to extract boundaries, this can take more than an hour!

If you used the commands above to get the data, then you can use the following command to start the training:
```sh
python train.py --root-imgs data/VOCdevkit/VOC2012 --root-segs data/trainval/ --run-path results --cuda
```
Where results is the folder where the checkpoints, tensorboard and other outputs will be saved.

# Inference
Once the training is finished, you can run the inference on your own images with:
```
python eval_cob.py --model-path results/checkpoints/cp_or.pth.tar --in-path in/ --out-path out/
```
