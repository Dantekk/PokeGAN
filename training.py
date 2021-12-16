import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from custom_callback import GANMonitor
from dataset import make_dataset
from settings import *
from utilities import load_img_dataset_into_list
from model import StyleGAN

dataset_img_folder = load_img_dataset_into_list(dataset_path)
print("Number of images : "+str(len(dataset_img_folder)))

START_RES = 4
TARGET_RES = 256

style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)

def train(start_res=START_RES, target_res=TARGET_RES, steps_per_epoch=5000):
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    val_batch_size = 16
    val_z = tf.random.normal((val_batch_size, style_gan.z_dim))
    val_noise = style_gan.generate_noise(val_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl = make_dataset(images=dataset_img_folder, res=res, data_aug=DATA_AUG_MODE)

            steps = steps_per_epoch

            style_gan.compile(
                d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )

            # Set checkpoint Callback
            ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
                f"checkpoints/stylegan_{res}x{res}_{phase}.ckpt",
                save_weights_only=True,
                verbose=0,
            )
            # set custom callback for test mode
            GANMon = GANMonitor(val_z, val_noise, res, phase)
            print(phase)

            style_gan.fit(train_dl, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb, GANMon])

# Start training
train(start_res=4, target_res=256, steps_per_epoch=8190)
