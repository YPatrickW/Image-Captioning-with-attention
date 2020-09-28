import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
original_image_dir = "./images/train2014/images/"
output_dir = "./images/resized_train/"



def resize_image(image, size):
    resized_image = image.resize(size, Image.ANTIALIAS)
    return resized_image


def resize_images(original_image_dir, output_dir, size):
    images_dir_list = os.listdir(original_image_dir)  # get a list to store image dirs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num_images = len(images_dir_list)
    image_resized = os.listdir(output_dir)
    for i, image_dir in enumerate(images_dir_list):
        with open(os.path.join(original_image_dir, image_dir), "rb+") as f:
            with Image.open(f) as image:
                image = resize_image(image, size)
                image.save(os.path.join(output_dir, image_dir), image.format)
        if (i + 1) % 1000 == 0:
            print("[{}/{}] has been resized in folder{}".format(i + 1, num_images, output_dir))


resize_images(original_image_dir, output_dir, size=[256,256])
