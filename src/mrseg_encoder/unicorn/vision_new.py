import os
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk
from mrsegmentator import config

config.disable_nnunet_path_warnings()
import sys

from mrseg_encoder.unicorn.patch_extraction import extract_patches
from mrseg_encoder.inference import encode 
from picai_prep.preprocessing import PreprocessingSettings, Sample
from tqdm import tqdm
from unicorn_baseline.io import resolve_image_path

DEBUG = False
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

# for debugging only
def init(input_path, output_path, model_path):
    global INPUT_PATH, OUTPUT_PATH, MODEL_PATH
    INPUT_PATH = Path(input_path)
    OUTPUT_PATH = Path(output_path)
    MODEL_PATH = Path(model_path)

def write_json_file(*, location, content):

    with open(location, "w") as f:
        json.dump(
            content,
            f,
            indent=4,
        )


def run(
    task_description: dict[str, str],
    input_information: list[dict[str, Any]],
    model_dir: Path,
    output_dir: Path,
) -> int:
    """
    Process input data
    """
    # retrieve task details
    domain = task_description["domain"]
    task_type = task_description["task_type"]
    task_name = task_description["task_name"]

    if domain == "pathology":
        raise ValueError(f"Domain '{domain}' not supported")

    elif (domain == "CT") | (domain == "MR"):
        run_radiology_vision_task(
            task_type=task_type,
            input_information=input_information,
            model_dir=model_dir,
            domain=domain,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Domain '{domain}' not supported yet for vision tasks.")

    return 0

def get_target_spacing(image: sitk.Image, target_size) -> list[float]:

    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_spacing = [
        (old_spacing[i] * old_size[i]) / target_size[i]
        for i in range(3)
    ]

    print("Image size:", old_size)
    print("Image spacing:",old_spacing)
    print("New spacing:", new_spacing)
    return new_spacing

def extract_features_segmentation(
    image,
    model_dir: str,
    domain: str,
    title: str = "patch-level-neural-representation",
    patch_size: list[int] = [64, 64,64],
    patch_spacing: list[float] | None = [1.0, 1.0,1.0],
    overlap_fraction: Iterable[float] = (0.0, 0.0, 0.0),
    compression_factor: int = 20,
    reduction_factor: int = 1,
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image
    """
    patch_features = []

    image_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )

    print(f"Extracting patches from image")
    patches, coordinates, image = extract_patches(
        image=image,
        patch_size=patch_size,
        spacing=patch_spacing,
        overlap_fraction=overlap_fraction,
    )
    if patch_spacing is None:
        patch_spacing = image.GetSpacing()

    print(f"Extracting features from patches")
    is_first = True
    for patch, coords in tqdm(
        zip(patches, coordinates), total=len(patches), desc="Extracting features", file=sys.stdout
    ):
         
        # only show verbose output for the first patch
        if is_first:
            embeddings = encode(patch,verbose=True, compression_factor=compression_factor)
            is_first = False
        else:
            embeddings = encode(patch,verbose=False, compression_factor=compression_factor)
        
        patch_features.append(
            {
                "coordinates": coords[0],
                "features": embeddings,
            }
        )

    patch_level_neural_representation = make_patch_level_neural_representation(
        patch_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=image.GetSize(),
        image_origin=image.GetOrigin(),
        image_spacing=image.GetSpacing(),
        image_direction=image.GetDirection(),
        title=title,
        reduction_factor=reduction_factor,
    )
    return patch_level_neural_representation


def make_patch_level_neural_representation(
    *,
    title: str,
    patch_features: Iterable[dict],
    patch_size: Iterable[int],
    patch_spacing: Iterable[float],
    image_size: Iterable[int],
    image_spacing: Iterable[float],
    image_origin: Iterable[float] = None,
    image_direction: Iterable[float] = None,
    reduction_factor: int = 1,
) -> dict:
    if image_origin is None:
        image_origin = [0.0] * len(image_size)
    if image_direction is None:
        image_direction = np.identity(len(image_size)).flatten().tolist()

    # reduce resolution for upsampling adaptor
    patch_size = [int(s//reduction_factor) for s in patch_size]
    patch_spacing = [s*reduction_factor for s in patch_spacing]

    return {
        "meta": {
            "patch-size": list(patch_size),
            "patch-spacing": list(patch_spacing),
            "image-size": list(image_size),
            "image-origin": list(image_origin),
            "image-spacing": list(image_spacing),
            "image-direction": list(image_direction),
        },
        "patches": list(patch_features),
        "title": title,
    }


def run_radiology_vision_task(
    *,
    task_type: str,
    input_information: dict[str, Any],
    model_dir: Path,
    domain: str,
    output_dir: str,
):
    # Identify image inputs
    image_inputs = []
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_inputs.append(input_socket)

    # T2 (single)
    if task_type == "classification":

        neural_representations = []
        image_representations = []
        for image_input in image_inputs:

            if DEBUG:
                image_dir = Path(str(INPUT_PATH) + str(image_input["input_location"]).replace("input/",""))
            else:
                image_dir = Path(image_input["input_location"])

            scan_path = next(image_dir.glob("*.mha"), None)

            if scan_path is None:
                continue

            print(f"Reading image from {scan_path}")
            image = sitk.ReadImage(str(scan_path))
            image = sitk.DICOMOrient(image, desiredCoordinateOrientation="LPS")
            embeddings = encode(image, verbose=True, compression_factor=16) # 80 features
            image_level_neural_representation = {
                "title": image_input["interface"]["slug"],
                "features": embeddings,
            }
            image_representations.append(image_level_neural_representation)
            continue

        output_path = output_dir / "patch-neural-representation.json"
        write_json_file(location=output_path, content=neural_representations)

        output_path = output_dir / "image-neural-representation.json"
        write_json_file(location=output_path, content=image_representations)

    elif task_type in ["detection", "segmentation"]:
        neural_representations = []

        if image_inputs[0]["interface"]["slug"].endswith("prostate-mri"):
            images_to_preprocess = {}
            titles = []
            for image_input in image_inputs:
                if DEBUG:
                    image_path = resolve_image_path(
                        location=Path(str(INPUT_PATH) + str(image_input["input_location"]).replace("input/",""))
                    )
                else:
                    image_path = resolve_image_path(location=image_input["input_location"])
                print(f"Reading image from {image_path}")
                image = sitk.ReadImage(str(image_path))

                if "t2" in str(image_input["input_location"]):
                    images_to_preprocess.update({"t2": image})
                if "hbv" in str(image_input["input_location"]):
                    images_to_preprocess.update({"hbv": image})
                if "adc" in str(image_input["input_location"]):
                    images_to_preprocess.update({"adc": image})
                titles.append(image_input["interface"]["slug"])

            pat_case = Sample(
                scans=[
                    images_to_preprocess.get("t2"),
                    images_to_preprocess.get("hbv"),
                    images_to_preprocess.get("adc"),
                ]
            )
            pat_case.preprocess()

            for title, image in zip(titles, pat_case.scans):
                neural_representation = extract_features_segmentation(
                    image=image,
                    model_dir=model_dir,
                    domain=domain,
                    title=title,
                    overlap_fraction=(0.5,0.5, 0.5),
                    compression_factor=1,  # feature length of 2560
                    reduction_factor=8,
                                                    )
                neural_representations.append(neural_representation)

        else:
            for image_input in image_inputs:

                if DEBUG:
                    image_path = resolve_image_path(
                        location=Path(str(INPUT_PATH) + str(image_input["input_location"]).replace("input/",""))
                    )
                else:
                    image_path = resolve_image_path(location=image_input["input_location"])
                print(f"Reading image from {image_path}")
                try:
                    image = sitk.ReadImage(str(image_path))
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
                    continue

                if task_type == "detection": # Task 7
                    neural_representation = extract_features_segmentation(
                        image=image,
                        model_dir=model_dir,
                        domain=domain,
                        title=image_input["interface"]["slug"],
                        overlap_fraction=(0.5,0.5,0.5),
                        compression_factor= 8, # feature length of 320
                        reduction_factor= 8
                    )
                elif task_type == "segmentation" and domain == "CT": # Task 10
                    neural_representation = extract_features_segmentation(
                        image=image,
                        model_dir=model_dir,
                        domain=domain,
                        title=image_input["interface"]["slug"],
                        overlap_fraction=(0.5,0.5,0.5),
                        compression_factor= 1, # feature length of 2560
                        reduction_factor= 16
                    )
                elif task_type == "segmentation" and domain == "MR": # Task 11
                    neural_representation = extract_features_segmentation(
                        image=image,
                        model_dir=model_dir,
                        domain=domain,
                        title=image_input["interface"]["slug"],
                        overlap_fraction=(0.5,0.5,0.5),
                        compression_factor= 1, # feature length of 2560
                        reduction_factor= 16
                    )
                else:
                    raise ValueError(f"Task type '{task_type}' not supported for domain '{domain}'.")


                neural_representations.append(neural_representation)

        output_path = output_dir / "patch-neural-representation.json"
        write_json_file(location=output_path, content=neural_representations)
