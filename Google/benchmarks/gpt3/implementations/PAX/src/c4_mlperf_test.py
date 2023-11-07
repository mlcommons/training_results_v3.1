"""referred to google3/platforms/deepsea/ffds/xor_bmk/bmk.py."""
import jax
from jax import numpy as jnp
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm.params import quant
from paxml.tasks.lm.params import quant_aqt_v2
from paxml.tasks.lm.params.gpt3 import C4SpmdGpt3AdamMLPerfHP
from paxml.tasks.lm.params.gpt3 import C4SpmdGpt3AdamPipeline
from praxis import pax_fiddle


@experiment_registry.register
class C4SpmdGpt3Adam2x8x4Test(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""
  NUM_LAYERS = 10
  NUM_HEADS = 24
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4

  PERCORE_BATCH_SIZE = 0.5
  ICI_MESH_SHAPE = [1, 8, 4]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 100

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  EVAL_INTERVAL_STEPS = 100
  EVAL_SKIP_TRAIN = True

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)
  TARGET_LOG_PPLX = 7.5

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 800

    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x4x4(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  NUM_LAYERS = 10
  NUM_HEADS = 24
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4

  PERCORE_BATCH_SIZE = 0.5
  ICI_MESH_SHAPE = [1, 4, 4]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  EVAL_INTERVAL_STEPS = 40
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 100  # Run for 80 steps
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel1x4x4(C4SpmdGpt3AdamDataParallel2x4x4):
  ICI_MESH_SHAPE = [1, 4, 4]
  DCN_MESH_SHAPE = [1, 1, 1]


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel1x2x2(C4SpmdGpt3AdamDataParallel2x4x4):
  ICI_MESH_SHAPE = [1, 2, 2]
  DCN_MESH_SHAPE = [1, 1, 1]
  NUM_LAYERS = 5


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel1x4x2(C4SpmdGpt3AdamDataParallel2x4x4):
  ICI_MESH_SHAPE = [1, 4, 2]
  DCN_MESH_SHAPE = [1, 1, 1]
  NUM_LAYERS = 5


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x8x8(C4SpmdGpt3AdamDataParallel2x4x4):
  ICI_MESH_SHAPE = [1, 8, 8]
  DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel1x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 128 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20
  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 192
  EVAL_SKIP_TRAIN = True

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.num_train_steps = 1000

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 256 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [2, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 96
  EVAL_SKIP_TRAIN = True
  LEARNING_RATE = 2.0e-5

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # only run 200 step ontop of initial ckp, won't reach target TARGET_LOG_PPLX
    task_p.train.num_train_steps = 24200
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel4x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 512 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [4, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 48
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # only run 200 step ontop of initial ckp, won't reach target TARGET_LOG_PPLX
    task_p.train.num_train_steps = 14000
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel8x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 1024 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [8, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 24
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    task_p.train.num_train_steps = 7000
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel12x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 1536 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [12, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 2
  EVAL_INTERVAL_STEPS = 16
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    task_p.train.num_train_steps = 7000
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel16x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [16, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 12
  EVAL_SKIP_TRAIN = True
  TRAINING_SEED = 3407

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    task_p.train.num_train_steps = 3400
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel16x16x16PerfOnly(
    C4SpmdGpt3AdamDataParallel16x16x16
):
  r"""Cross-slice data-parallel GPT-3 config."""

  EVAL_INTERVAL_STEPS = 1000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0
    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    task_p.train.num_train_steps = 120
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel16x16x16CrossSliceWS(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.5  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [1, 16, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 12
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN

    task_p.train.num_train_steps = 3000
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel32x16x16(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.25  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [32, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 12
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN

    task_p.train.num_train_steps = 3000
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel32x16x16CrossSliceWS(C4SpmdGpt3AdamMLPerfHP):
  r"""Cross-slice data-parallel GPT-3 config."""

  PERCORE_BATCH_SIZE = 0.25  # 2048 global batch size
  ICI_MESH_SHAPE = [1, 16, 16]
  DCN_MESH_SHAPE = [1, 32, 1]
  FPROP_DTYPE = jnp.bfloat16

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  PERCORE_EVAL_BATCH_SIZE = 1.5
  EVAL_INTERVAL_STEPS = 2
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.summary_verbosity = 0

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN

    task_p.train.num_train_steps = 3000
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x16x16Int8(C4SpmdGpt3AdamDataParallel2x16x16):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel2x16x16Int8V1(
    C4SpmdGpt3AdamDataParallel2x16x16
):
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant.apply_quantized_layers_sharded(model_p, quant.F8B8)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline4x16x16(C4SpmdGpt3AdamPipeline):
  """GPT3 on 4 VLP256 pipeline for convergence test.

  # pylint: disable=g-line-too-long
  reference to https://source.corp.google.com/piper///depot/google3/platforms/deepsea/ffds/xor_bmk/bmk.py;l=751
  """

  PERCORE_BATCH_SIZE = 2  # 2048 global batch size

  NUM_STAGES = 4
  ICI_MESH_SHAPE = [1, 1, 16, 16]
  DCN_MESH_SHAPE = [4, 1, 1, 1]
  FPROP_DTYPE = jnp.bfloat16

  EMB_W_DATA_DIMS = ("replica", "data")
  STREAM_IO = True
  USE_REPEATED_LAYER = False

  MICROBATCH_SIZE = 32  # 64 microbatches
  SUMMARY_INTERVAL_STEPS = 12

  # for mlperf_gpt3_checkpoint_every_n_steps=2000
  CHECKPOINT_EVERY_N_STEPS = 2000

  # mlperf_gpt3_checkpoint_max_to_keep=20
  CHECKPOINT_SAVE_MAX_TO_KEEP = 20

  # for jax_softmax_custom_jvp
  jax.config.update("jax_softmax_custom_jvp", False)

  EVAL_INTERVAL_STEPS = 12
  EVAL_SKIP_TRAIN = True

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 3000

    task_p.train.eval_skip_train = self.EVAL_SKIP_TRAIN
    # mlperf_gpt3_summary_verbosity=0
    task_p.summary_verbosity = 0

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamPipeline4x16x16Int8(C4SpmdGpt3AdamPipeline4x16x16):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel8x16x16Int8(C4SpmdGpt3AdamDataParallel8x16x16):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel12x16x16Int8(
    C4SpmdGpt3AdamDataParallel12x16x16
):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel16x16x16Int8(
    C4SpmdGpt3AdamDataParallel16x16x16
):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel16x16x16Int8V1(
    C4SpmdGpt3AdamDataParallel16x16x16
):
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant.apply_quantized_layers_sharded(model_p, quant.F8B8)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel12x16x16Int8PerfOnly(
    C4SpmdGpt3AdamDataParallel12x16x16Int8
):
  EVAL_INTERVAL_STEPS = 1000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    task_p.train.num_train_steps = 110
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel16x16x16Int8PerfOnly(
    C4SpmdGpt3AdamDataParallel16x16x16Int8
):
  EVAL_INTERVAL_STEPS = 1000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    task_p.train.num_train_steps = 120
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel32x16x16Int8(
    C4SpmdGpt3AdamDataParallel32x16x16
):
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant_aqt_v2.apply_quantized_layers_sharded(model_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamDataParallel32x16x16Int8V1(
    C4SpmdGpt3AdamDataParallel32x16x16
):
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    quant.apply_quantized_layers_sharded(model_p, quant.F8B8)
    return task_p
