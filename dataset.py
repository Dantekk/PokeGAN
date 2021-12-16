import tensorflow as tf
from settings import *
AUTOTUNE = tf.data.AUTOTUNE
from image_aug import *

def read_image(data_aug, res):
    def decode_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        #img = tf.image.resize(img, IMAGE_SIZE)

        if data_aug:
            img = augment(img)

        # only donwsampling, so use nearest neighbor that is faster to run
        img = tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img

    def augment(img):
        img = tf.expand_dims(img, axis=0)
        img = img_transf(img)
        img = tf.squeeze(img, axis=0)
        return img

    return decode_image

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

def make_dataset(images, res, data_aug):
    read_image_xx = read_image(data_aug, res)
    img_dataset = tf.data.Dataset.from_tensor_slices(images)

    img_dataset = (img_dataset
                   .map(read_image_xx, num_parallel_calls=AUTOTUNE))

    dataset = img_dataset.shuffle(SHUFFLE_DIM).batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE).repeat()
    return dataset
