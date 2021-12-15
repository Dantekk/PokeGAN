# PokeGAN
A tensorflow/keras implementation of StyleGAN to generate images of new Pokemon.
## Dataset
The model has been trained on dataset that includes 819 pokémon. </br>
You can download dataset from [this kaggle link](https://www.kaggle.com/kvpratama/pokemon-images-dataset).
## Dependencies
I have used the following versions for code work:
* python==3.8.8
* tensorflow==2.4.1
* tensorflow-gpu==2.4.1
* numpy==1.19.1
* h5py==2.10.0
## Note
There are several difficulties in pokemon generation using GAN : </br>
* The difficulty of GAN training is well known; changing a hyperparameter can greatly change the results.
* **The dataset size is too small**! 819 different pokemon images are not enough. For this reason, I applied **data augmentation on the data**; these are the transformations applied : 
```python
img_transf = tf.keras.Sequential([
            	tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.05, 0.15)),
                image_aug.RandomBrightness(brightness_delta=(-0.15, 0.15)),
                image_aug.PowerLawTransform(gamma=(0.8,1.2)),
                image_aug.RandomSaturation(sat=(0, 2)),
                image_aug.RandomHue(hue=(0, 0.15)),
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
	    	tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
		tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
		tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.10, 0.10))])
```
* StyleGAN training is very expensive! I trained the model starting from a 4x4 resolution up to the **final resolution of 256x256**. The model was **trained for 8 days using a Tesla V100 32GB SXM2**. </br>To get even better results you need to use higher resolutions and train for longer time.
# Results
These are an example of new pokémon :

## More results
You can see hundreds of new pokemon in __ folder. </br>
I repeat again : **to get better results (better details in pokemon) it is necessary to train for more time**.


## References
This code implementation is inspired by the unofficial keras implementation of styleGAN.