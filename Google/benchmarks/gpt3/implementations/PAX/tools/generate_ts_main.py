"""Script to generate simple array and store as TensorStore format.
"""

from absl import app
from absl import flags
import numpy as np
import tensorstore as ts

FLAGS = flags.FLAGS

_VALUE = flags.DEFINE_integer(
    "value",
    0,
    "Value to fill the array.")

_SHAPE = flags.DEFINE_string(
    "shape",
    None,
    "Shape of the array. E.g. \"8,8\".")

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "Output path.")


def main(_) -> None:
  shape_str = _SHAPE.value
  shape = []
  if shape_str:
    for token in shape_str.split(","):
      shape.append(int(token))

  a = _VALUE.value * np.ones(shape, dtype=np.int32)
  output_path = _OUTPUT_PATH.value
  print(
      "output_path = ", output_path, ", type = ", a.dtype, ", shape = ", a.shape
  )
  print("a = ", a)

  spec = {"driver": "zarr", "kvstore": {}, "metadata": {}}
  spec["kvstore"] = {
      "driver": "file",
      "path": output_path,
  }
  t = ts.open(
      ts.Spec(spec), open=True, create=True,
      dtype=a.dtype, shape=a.shape
  ).result()
  f = t.write(a).commit
  f.result()


if __name__ == "__main__":
  app.run(main)
