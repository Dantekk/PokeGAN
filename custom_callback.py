import tensorflow as tf
import matplotlib.pyplot as plt
from settings import BATCH_SIZE

def generate_and_save_images(images, batch, res, phase):
    grid_col = 4
    grid_row = 4

    f, axarr = plt.subplots(
        grid_row, grid_col, figsize=(res, res)
    )

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis("off")

    f.savefig("output_figs/{}_{}_{}".format(res, phase, batch))

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, val_z, val_noise, res, phase):
        self.val_z = val_z
        self.val_noise = val_noise
        self.res = res
        self.phase = phase

    def on_train_batch_end(self, batch, logs=None):
        if batch % (BATCH_SIZE*5) == 0:
            images = self.model({"z": self.val_z, "noise": self.val_noise, "alpha": 1.0})
            generate_and_save_images(images, batch, self.res, self.phase)
