#Colorization
By *[nilboy](https://github.com/nilboy)*

A Tensorflow implementation of ECCV2016 paper([Colorful Image Colorization](https://arxiv.org/pdf/1603.08511.pdf))

![presentation](https://raw.githubusercontent.com/nilboy/colorization-tf/master/resources/display.jpg)

###Train

##### Download imagenet data

1. Download imagenet data and Extract to one directory named 'Imagenet'

2. link the Imagenet data directory to this project path

	```
	ln -s $Imagenet data/imagenet
	```

#### convert the imagenet data to text_record file

```
python tools/create_imagenet_list.py
```

#### train

```
python tools/train.py -c conf/train.cfg
```

#### Train your customer data

1. transform your training data to text_record file

2. calculate the training data prior-probs(reference to tools/create_prior_probs.py)

3. write your own train-configure file

4. train (python tools/train.py -c $your_configure_file)

###test demo

1. Download pretrained model(<a>https://drive.google.com/file/d/0B-yiAeTLLamRWVVDQ1VmZ3BxWG8/view?usp=sharing</a>)

	```
	mv color_model.ckpt models/model.ckpt
	```
2. Test

	```
	python demo.py
	```


