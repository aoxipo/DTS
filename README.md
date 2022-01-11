# DTS
DTS is a neural network based on  residual Unet(hourglass) + Map attention + transformer(detr)   it's be applied in the star detection

Our model consists of two parts ，it's Residual Unet front net and transformer backend net.  we  combination Residual  Unet and transformer with Map attention. 

the experiment result shows that Our model converges over the first 20 epochs, Our **MAP** on the training set and test set reached **93.737%** and **92.337%** respectively 。finally We achieved **97.20%** accuracy and **1.56** loss on the test set 。



## Train

our model consists of Residual Unet and transformer net.



#### Residual Unet Train

```shell
python hourglass_train.py
```

In this part we set the learning rate is 1e-5 and set 100 and 500 epoch ,gained 0.132 loss and 0.11 loss(MSE) 



#### Detr Train 

```shell
python DETR_train.py
```

In this part we set the learning rate is 1e-5 and set 100 epoch ,gained 97.533% and 97.209% accuracy on train and test set. our detr net define as :

```
trainer = Train(
	hidden_dim = 512,
    dropout = 0.1,
    nheads = 8,
    dim_feedforward = 2048,
    enc_layers = 6,
    dec_layers = 6,
    pre_norm = True,
    is_show = False,
    class_number = 1, 
    require_number = 10,
    image_shape = (64,64)
)
```

You need at least 15GB RAM to Train dataset and 3GB to Val dataset . Our experiment use 3090(24GB)



#### Test

set  trainer.test(100, val_dataloader) in the 432 line and run 

```
python DETR_train.py
```

before your start run the command please check your model parameter is right.



## DataSet

we make the dataset by data_generate.py and its use slide window to crop the image

we provide four different scale image dataset. "LL, L, M, S"  it's mean the different step for slide window.
