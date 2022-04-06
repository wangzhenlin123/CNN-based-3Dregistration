# A CNN based 3D multi-modal unsupervised rigid registration network

Recently, many non-rigid registration network need rigid registration as a neccesary data preprocessing part. This project provides an easy to use, 
efficient model to do rigid registration. We can use this modal to do data preprocessing, or connect to a non-rigid network as a subnet.

## Network Structure

<img width="512" alt="image" src="https://user-images.githubusercontent.com/52573031/162075020-07f4baf7-457c-4f65-99be-515e69de5efb.png">

## Result

<img width="512" alt="image" src="https://user-images.githubusercontent.com/52573031/162075223-5578e959-1dd7-46a5-8475-18d30eeb7ebb.png">

## Config
```
{
  "if_load": false,    #if you want to use pre-trained model, set "true"
  "lr": 1e-05,    #learning rate
  "inshape": [128, 128, 128],    #input shape
  "input_channels": 2,      #sum of channels of fixed image and moving image
  "epoch": 300,     #train epoch
  "batch_size": 16,     #batch size
  "train_iteration": 50,     #train iteration in each epoch (depend on your data provider)
  "test_iteration": 50,      #test iteration in each epoch (depend on your data provider)
  "save_epoch": 20,          #save model in every 20 epochs
  "save_path": "./save".     #where to save/load model
}
```
## How to use
### 1. Do preprocessing.
Don't need much tricky preprocessing, but commonly used methods are highly recommended. E.g. clipping pixel value, standardization, normalization. Make 
sure that moving image and fixed image have the same shape.
The recommended shape of input image is [128, 128, 128], if you want to use other input shape, maybe you need to add/delete some convolutional layers
to make sure that after convolutional stage, image information is transformed to channel information. If you don't want to change network structure, 
you can interpolate/downsample your image to [128, 128, 128].

### 2. Write your own data provider.
Write data provider, then edit main.train() to use your data provider. The output shape of the data provider should be ((N,C,D,H,W),(N,C,D,H,W))

### 3. Run train.py
Run train.py and wait for training.
