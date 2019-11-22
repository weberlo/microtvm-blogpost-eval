# first `pip install efficientnet`

from efficientnet import inject_keras_modules
from efficientnet.model import EfficientNet

from tvm.relay.frontend import from_keras

def micronet(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    return EfficientNet(0.1, 0.1, 224, 0.2,
                        model_name='efficientnet-b0',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)
keras_model = inject_keras_modules(micronet)
keras_model = keras_model()

in_shapes = []
for layer in keras_model._input_layers:
    in_shapes.append(tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape))
shape_dict = {name: shape for (name, shape) in zip(keras_model.input_names, in_shapes)}

mod = from_keras(keras_model, shape_dict)
print(mod)
