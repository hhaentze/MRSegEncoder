# Copyright 2024 Hartmut Häntze

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def initialize() -> Any:
    name = "MRSeg-Encoder"
    desc = "Encoder-Only Version of MRSegmentator"
    epilog = "AIAH Lab - 2025"

    parser = argparse.ArgumentParser(prog=name, description=desc, epilog=epilog)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to input image",
    )

    parser.add_argument("--outdir", type=str, default=".", help="output directory")

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="choose a model based on the validation folds",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("--cpu_only", action="store_true", help="don't use a gpu")

    args = parser.parse_args()
    return args


def assert_namespace(namespace: Any) -> None:
    # requirements

    assert os.path.isdir(
        Path(namespace.outdir).parent
    ), f"Parent of output directory {namespace.outdir} not found"
    assert os.path.isfile(namespace.input), f"Input {namespace.input} not found"
