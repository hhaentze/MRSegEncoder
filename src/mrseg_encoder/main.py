# Copyright 2025 Hartmut Häntze

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from datetime import timedelta

import torch
from mrseg_encoder import parser, utils
from mrsegmentator import config

config.disable_nnunet_path_warnings()

from mrseg_encoder.inference import infer  # noqa: E402


def main() -> None:
    # initialize Parser
    namespace = parser.initialize()
    parser.assert_namespace(namespace)

    # select images for segmentation
    images = utils.read_images(namespace)
    image = images[0]

    start_time = time.time()
    # run inference
    infer(
        image,
        namespace.outdir,
        0,
        namespace.verbose,
        namespace.cpu_only,
    )
    end_time = time.time()
    time_delta = timedelta(seconds=round(end_time - start_time))
    print(f"Finished encoding in {time_delta}.")


def encode_path(img_path, outdir=None, is_pathology_wsi=False):

    start_time = time.time()
    # run inference
    results = infer(
        img_path,
        outdir,
        0,
        False,
        not torch.cuda.is_available(),
        return_results=True if outdir is None else False,
        is_pathology_wsi=is_pathology_wsi,
    )
    end_time = time.time()
    time_delta = timedelta(seconds=round(end_time - start_time))
    print(f"Finished encoding in {time_delta}.")

    if outdir is None:
        return results


if __name__ == "__main__":
    main()
