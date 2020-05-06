import argparse
import base64
import hashlib
import glob
import json
import pickle
import random
import sys
import tarfile
import tempfile
import time

import numpy
import requests

from micro_eval.util import cifar10_data


# Number of classes
NUM_CLASSES = 10


# Number of images per cifar-10 class in the source dataset.
NUM_IMAGES_PER_CLASS = 6000


# Number of images to save to disk per cifar-10 class.
TARGET_NUM_IMAGES_PER_CLASS = 50


CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'


def main():
  md5 = hashlib.md5()
  with tempfile.TemporaryDirectory() as tempdir:
#    print('Downloading cifar10 dataset...')
    # with requests.get('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', stream=True) as r:
    #   tar_path = f'{tempdir}/cifar10.tar.gz'
    #   with open(tar_path, 'wb') as tar_f:
    #     for i, chunk in enumerate(r.iter_content(10240)):
    #       md5.update(chunk)
    #       tar_f.write(chunk)
    #       if i % 128 == 0:
    #         sys.stderr.write('.')
    #         sys.stderr.flush()

    #     sys.stderr.write('\n')

    #   if md5.hexdigest() != CIFAR10_MD5:
    #     print(f'md5 mismatch: want {CIFAR10_MD5}, got {md5.hexdigest()}')
    #     sys.exit(2)

      tar_path = f'cifar-10-python.tar.gz'
      print('Extracting...')
      with tarfile.open(tar_path, 'r:gz') as tar_f:
        tar_f.extractall(tempdir)

      print(f'Choosing {TARGET_NUM_IMAGES_PER_CLASS} samples per class...')
      to_extract_by_class = [random.sample(range(NUM_IMAGES_PER_CLASS), TARGET_NUM_IMAGES_PER_CLASS)
                             for _ in range(NUM_CLASSES)]
      last_image_index = [-1] * NUM_CLASSES
      prob = TARGET_NUM_IMAGES_PER_CLASS / NUM_IMAGES_PER_CLASS  # Sample this many images
      random.seed(time.time())
      arr = []

      print(f'Loading data from {tempdir}/cifar-10-batches-py/data_batch_*')
      for path in glob.glob(f'{tempdir}/cifar-10-batches-py/data_batch_*'):
        print(f'looking {path}')
        with open(path, 'rb') as pickle_f:
          obj = pickle.load(pickle_f, encoding='bytes')
          for label, filename, image in zip(obj[b'labels'], obj[b'filenames'], obj[b'data']):
            img_index = last_image_index[label] + 1
            last_image_index[label] = img_index

            if last_image_index[label] not in to_extract_by_class[label]:
              continue

            arr.append(dict(label=label, filename=str(filename, 'utf-8'), image=str(base64.b85encode(image), 'utf-8')))

      with open(cifar10_data.DATA_FILE_PATH, 'w') as out_f:
        json.dump(arr, out_f)


if __name__ == '__main__':
  main()
