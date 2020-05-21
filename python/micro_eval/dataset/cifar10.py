import mxnet
import numpy

from . import DatasetGenerator, DatasetSample
from micro_eval import util


class Cifar10DatasetGenerator(DatasetGenerator):

  DATA_LAYOUT = 'NHWC'

  def generate(self, num_samples):
    """Grabs input/label pairs from MNIST"""
    ctx = mxnet.cpu()
    # Load a random image from the test dataset
    sample_data = mxnet.gluon.data.DataLoader(
            mxnet.gluon.data.vision.CIFAR10(train=False),
            1,
            shuffle=self.config.get('shuffle', False))

    samples = []
    for i, (data, label) in zip(range(num_samples), sample_data):
        if i == num_samples:
            break

        data_np = numpy.copy(data.as_in_context(ctx).asnumpy())
        # gluon data is in NHWC format
        shape = util.LabelledShape(dim_iter=zip('NHWC', data_np.shape), dtype='uint8')
        data_nt = util.LabelledTensor(data_np, shape).with_layout(self.DATA_LAYOUT)

        label = int(label.asnumpy()[0])
        samples.append(DatasetSample(inputs={'data': data_nt},
                                     outputs={'label': label}))

    return samples
