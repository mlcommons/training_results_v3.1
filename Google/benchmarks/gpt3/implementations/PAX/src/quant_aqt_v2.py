"""Wrapper classes for enabling quantization during training."""
from absl import flags
from absl import logging
from aqt.jax.v2 import aqt_dot_general
from aqt.jax.v2 import config
from fiddle import selectors
import jax.numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis.layers import transformers


_MLPERF_GPT3_AQT_CONFIG_NAME = flags.DEFINE_enum(
    'mlperf_gpt3_aqt_config_name',
    default='ttt',
    enum_values=['ttf', 'ttt'],
    help='accumulator dtype',
)

_MLPERF_GPT3_LOCAL_AQT_factor = flags.DEFINE_integer(
    'mlperf_gpt_local_aqt_factor',
    default=None,
    help='local_aqt_factor',
)


def get_aqt_config(
    aqt_cfg_name: str = None,
    local_aqt_factor: int = None,
) -> config.DotGeneral:
  """return aqt config."""

  assert aqt_cfg_name is not None
  if aqt_cfg_name in ('ttf_drhs', 'ttf_dlhs'):
    fwd = config.DotGeneralRaw.make(8, 8)
    if aqt_cfg_name == 'ttf_drhs':
      dlhs = config.DotGeneralRaw.make(8, 8)
      drhs = config.DotGeneralRaw.make(None, None)
    else:  # aqt_cfg_name = 'ttf_dlhs'
      dlhs = config.DotGeneralRaw.make(None, None)
      drhs = config.DotGeneralRaw.make(8, 8)

    aqt_cfg = config.DotGeneral(fwd=fwd, dlhs=dlhs, drhs=drhs)

    # Surprising: lhs quantization determines what drhs can do.
    # Only rhs is accepting MultiTensor.
    aqt_cfg.drhs.rhs.use_fwd_quant = False
    aqt_cfg.dlhs.rhs.use_fwd_quant = False
    config.set_stochastic_rounding(
        aqt_cfg,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        implementation='custom-1'
    )
  elif aqt_cfg_name in ('ttt_drhs', 'ttt_dlhs'):
    # ttt_drhs and ttt_dlhs are equivalent in the current implementation
    # which would be different with localization in the future
    if aqt_cfg_name == 'ttt_drhs':
      dlhs_local_aqt_factor = None
      drhs_local_aqt_factor = local_aqt_factor
    else:  # aqt_cfg_name == "ttt_dlhs"
      dlhs_local_aqt_factor = local_aqt_factor
      drhs_local_aqt_factor = None

    aqt_cfg = config.fully_quantized(
        fwd_bits=8,
        bwd_bits=8,
        use_fwd_quant=False,
        use_stochastic_rounding=None,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        use_dummy_static_bound=False,
        dlhs_local_aqt_factor=dlhs_local_aqt_factor,
        drhs_local_aqt_factor=drhs_local_aqt_factor,
    )
    config.set_stochastic_rounding(
        aqt_cfg,
        vjp_lhs_stochastic_rounding=True,
        vjp_rhs_stochastic_rounding=False,
        implementation='custom-1'
    )

    accumulator_dtype = jnp.int32
    config.set_accumulator_dtype(
        aqt_cfg,
        fwd_dtype=accumulator_dtype,
        bwd_dtype=accumulator_dtype,
    )
  else:
    raise ValueError(f'Unsupported aqt_cfg_name: {aqt_cfg_name}')

  logging.info('>>> AQT(%s) config: %s', aqt_cfg_name, aqt_cfg)
  if local_aqt_factor:
    logging.info(
        '>>> AQT config(%s) using local_aqt_factor: %d',
        aqt_cfg_name,
        local_aqt_factor,
    )
  else:
    logging.info('>>> AQT config(%s) Not using local_aqt_factor', aqt_cfg_name)

  return aqt_cfg


class DqQuantEinsum(base_layer.BaseLayer):
  """Einsum layer with quantization."""

  aqt_cfg: config.DotGeneral | None = None

  def __call__(self, eq, lhs, rhs):
    if not self.aqt_cfg:
      raise ValueError(f'Empty aqt_cfg in {self.name}.')
    aqt_key = self.next_prng_key()
    def dg(lhs, rhs, axes, precision=None, preferred_element_type=None):
      del precision, preferred_element_type

      # Stochastic rounding in applied only to the gradient tensor i.e. lhs
      dot = aqt_dot_general.make_dot_general(self.aqt_cfg)
      context = aqt_dot_general.Context(key=aqt_key, train_step=None)

      return dot(lhs, rhs, axes, context=context)

    return jnp.einsum(eq, lhs, rhs, _dot_general=dg)


def apply_quantized_layers_sharded(model):
  """Adds quantization to existing transformer layers."""

  einsum_tpl_drhs = pax_fiddle.Config(
      DqQuantEinsum,
      aqt_cfg=get_aqt_config(
          aqt_cfg_name=f'{_MLPERF_GPT3_AQT_CONFIG_NAME.value}_drhs',
          local_aqt_factor=_MLPERF_GPT3_LOCAL_AQT_factor.value,
      ),
  )
  einsum_tpl_dlhs = pax_fiddle.Config(
      DqQuantEinsum,
      aqt_cfg=get_aqt_config(
          aqt_cfg_name=f'{_MLPERF_GPT3_AQT_CONFIG_NAME.value}_dlhs',
          local_aqt_factor=_MLPERF_GPT3_LOCAL_AQT_factor.value,
      ),
  )

  if (
      _MLPERF_GPT3_AQT_CONFIG_NAME.value == 'ttf'
      and _MLPERF_GPT3_LOCAL_AQT_factor.value
  ):
    raise ValueError(
        'TTF should not need local_aqt_factor at all,'
        ' _MLPERF_GPT3_LOCAL_AQT_factor should not be set'
    )

  if hasattr(model, 'lm_tpl'):
    logging.info('quantize attention: QKV')
    # Quantize attention: QKV
    selectors.select(model, layers.attentions.CombinedQKVProjectionLayer).set(
        einsum_tpl=einsum_tpl_dlhs
    )
    logging.info('quantize attention projection')
    # Quantize attention projection
    # use drhs for fused QKV attention where activations and weights are swapped
    selectors.select(model, layers.attentions.AttentionProjection).set(
        einsum_tpl=einsum_tpl_drhs
    )
    # logging.info('quantize attention output projection')
    # # Quantize attention output projection
    # selectors.select(model, layers.attentions.DotProductAttention).set(
    #     qk_einsum_tpl=einsum_tpl,
    #     pv_einsum_tpl=einsum_tpl,
    # )
    logging.info('quantize feedforward layers')
    # Quantize feedforward layers.
    xformer_p = model.lm_tpl.stacked_transformer_tpl
    if xformer_p.cls == transformers.PipelinedTransformer:
      xformer_p = xformer_p.pipeline_stage

    if xformer_p.cls == transformers.StackedTransformerRepeated:
      xformer_p = xformer_p.block
    xformer_p = xformer_p.transformer_layer_params_tpl
    xformer_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.set(
        einsum_tpl=einsum_tpl_drhs
    )
