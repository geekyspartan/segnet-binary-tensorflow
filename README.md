# SegNet
SegNet is a TensorFlow implementation of the [segmentation network proposed by Kendall et al.](http://mi.eng.cam.ac.uk/projects/segnet/), with cool features
like strided deconvolution, a minified architecture and more.

## Configuration
Create a `config.py` file, containing color maps, working dataset and other options.

```
autoencoder = 'segnet'
colors = {
  'segnet-32': [
  	[0, 0, 0],
    [255, 255, 255]
  ]
}
gpu_memory_fraction = 0.7
strided = True
```

Two kinds of architectures are supported at the moment: the original SegNet
Encoder-Decoder (`segnet`), and a smaller version of the same (`mini`), which
can be used for simpler segmentation problems. I suggest to use `strided = True`
for faster and more reliable results.

The `dataset_name` needs to match the data directories you create in your `input` folder.

## Train and test
Generate your TFRecords using `tfrecorder.py`. In order to do so, put your PNG
images in a `raw` folder, as follows:

```
input/
    raw/
        train/
        train-labels/
        test/
        test-labels/

```

Once you have your TFRecords, train SegNet with `python src/train.py`. Analogously, test it with `python src/test.py`.
