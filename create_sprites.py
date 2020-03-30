import numpy as np
from scipy import misc
from PIL import Image
import cv2
import os

IMAGE_SIZE = 40


def images_to_sprite(data):
    """
    Creates the sprite image
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (
        data.ndim - 3
    )
    data = np.pad(data, padding, mode="constant", constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
    )
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


if __name__ == "__main__":
    data = []
    INPUT_IMAGE_DIR = "saved_faces"
    for username in os.listdir(INPUT_IMAGE_DIR):
        if not os.path.isdir(os.path.join(INPUT_IMAGE_DIR, username)):
            continue

        embeddings = np.empty([0, 512])
        for f in os.listdir(os.path.join(INPUT_IMAGE_DIR, username)):
            if ".jpg" not in f:
                continue

            img_path = os.path.join(INPUT_IMAGE_DIR, username, f)
            image = cv2.imread(img_path)
            img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            nimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(nimg)
    img_sprite = images_to_sprite(np.array(data))
    sprite = Image.fromarray(img_sprite.astype(np.uint8))
    sprite.save("oss_data/sprites.png")
