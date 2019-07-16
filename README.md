# Torch-BCMNet
Classifier utilizing geometric correlation score between two images

## Description
+ From 'Longitudinal Change Detection on Chest X-rays using Geometric Correlation Maps (2019, MICCAI)'
+ Compare initial and followup chest x-rays to find out changes within certain temporal distances
+ Modified above model and adapted it to bilateral maxillary sinusitis diagnosis problem


## Files
+ dataloader.py : prepare and augment dataset
+ main.py : train and evaluate models
+ bcmnet.py : bilateral correlation map based network using resnet as backbone model
