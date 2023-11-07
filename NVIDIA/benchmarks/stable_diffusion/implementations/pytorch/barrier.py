# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from common import simple_init_distributed, barrier


def print0(*args, **kwds):
    if int(os.environ["SLURM_PROCID"]) == 0:
        print(*args, **kwds)


def main():
    print0("Before barrier")
    simple_init_distributed()
    barrier()
    print0("After barrier")


if __name__ == "__main__":
    main()
