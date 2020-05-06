import argparse
from micro_eval.model import cifar10_cnn


VALID_CONV_OP_IMPLS = ('direct', 'direct_simd')


def define_cifar10_conv_op_impl(parser):
  def _validate_conv_op_impls(spec):
    if ',' not in spec:
      spec = [spec.strip()] * 3
    else:
      spec = [s.strip() for s in spec.split(',')]
      if len(spec) != 3:
        raise argparse.ArgumentTypeError('expected 3 parts, got: %r' % (spec,))

    for impl in spec:
      if impl not in VALID_CONV_OP_IMPLS:
        raise argparse.ArgumentTypeError('expected one of %s, got: %s' % (','.join(VALID_CONV_OP_IMPLS), impl))

    return spec

  parser.add_argument('--cifar10-conv-op-impl', default=_validate_conv_op_impls('direct_simd'),
                      type=_validate_conv_op_impls,
                      help=('Conv2d op impls to use for cifar10 network. Choices are "direct_simd" '
                            'and "direct." Can name either a single choice (use for all 3 conv2d '
                            'ops) or a comma-separated list of 3'))


def has_simd_strategy(cifar10_conv_op_impl):
  return any(s == 'direct_simd' for s in cifar10_conv_op_impl)


# Kernel layouts, keyed by op implementation name. Valid for micro_dev target only, on x86 defaults
# to HWIO due to broader support for data layouts.
KERNEL_LAYOUTS = {
    'direct': 'HWIO',
    'direct_simd': 'HWOI',
}


BUILD_RELAY_MOD_RETURN = None


def build_relay_mod(cifar10_conv_op_impl, target, use_random_params=True):
    # per-conv op strategies (first entry is the strategy of the first conv and so on).
    # we want the ability to configure the op strategy, instead of just using
    # the best strategy in the log, because certain strategy combos have a
    # memory footprint that exceeds the available memory of the device.
    data_layout = 'NHWC'
    kernel_layouts = []
    for strat in cifar10_conv_op_impl:
      #            assert DATA_LAYOUTS[strat] == data_layout, 'data layouts for all convs must agree'
      if target == 'x86':
        kernel_layouts.append('HWIO')
      else:
        kernel_layouts.append(KERNEL_LAYOUTS[strat])

    _has_simd_strategy = has_simd_strategy(cifar10_conv_op_impl)
    op_strategy = 'direct_simd' if _has_simd_strategy else 'direct'
    mod, params = cifar10_cnn.gen_cifar10_cnn(
      data_layout, kernel_layouts,
      op_strategy=op_strategy,
      use_random_params=use_random_params)

    BUILD_RELAY_MOD_RETURN = argparse.Namespace(
      data_layout=data_layout,
      has_simd_strategy=_has_simd_strategy,
      mod=mod,
      params=params,
    )

    return BUILD_RELAY_MOD_RETURN
