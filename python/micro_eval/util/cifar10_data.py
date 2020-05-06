import collections
import json
import os
import sys

from micro_eval import util

import pandas


CIFAR10_LABELS = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


Entry = collections.namedtuple('Entry', 'filename label label_text image')


DATA_FILE_PATH = f'{util.get_repo_root()}/data/cifar10-validation.json'


def gen_validation_set(num_samples, seed=42):
  if not os.path.exists(DATA_FILE_PATH):
    raise Exception(f'CIFAR10 validation data not downloaded; run micro_eval/util/populate_cifar10_validation_data.py')

  with open(DATA_FILE_PATH) as json_f:
    obj = json.load(json_f)

  rand = random.Random(seed)
  to_return = rand.sample(len(data), num_samples)

  data = []
  for i in to_return:
    data.append(Entry(filename=obj[i]['filename'],
                      label=obj[i]['label'],
                      label_text=CIFAR10_LABELS[obj[i]['label']],
                      image=numpy.frombuffer(base64.b85decode(obj[i]['image']), shape=(25, 25))))

  return data
