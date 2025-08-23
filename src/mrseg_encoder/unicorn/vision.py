import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import SimpleITK as sitk
from mrsegmentator import config

config.disable_nnunet_path_warnings()
import sys

from mrseg_encoder.main import encode_path
from mrseg_encoder.unicorn.patch_extraction import extract_patches
from mrseg_encoder.inference import encode 
from picai_prep.preprocessing import PreprocessingSettings, Sample
from tqdm import tqdm
from unicorn_baseline.io import resolve_image_path


DEBUG = False

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


def extract_features_segmentation(
    image,
    model_dir: str,
    domain: str,
    title: str = "patch-level-neural-representation",
    # patch_size: list[int] = [16, 64, 64],
    patch_size=[64, 64, 16],
    patch_spacing: list[float] | None = [1.0, 1.0,1.0],
    overlap_fraction: Iterable[float] = (0.0, 0.0, 0.0),
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image
    """
    patch_features = []

    image_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )
    # if (image_orientation != "SPL") and (domain == "CT"):
    #     image = sitk.DICOMOrient(image, desiredCoordinateOrientation="SPL")

    if image_orientation != "LPS":
        image = sitk.DICOMOrient(image, desiredCoordinateOrientation="LPS")

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
    for patch, coords in tqdm(
        zip(patches, coordinates), total=len(patches), desc="Extracting features", file=sys.stdout
    ):
         
        embeddings = encode(patch, verbose=True)
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
    )
    return patch_level_neural_representation

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
) -> dict:
    if image_origin is None:
        image_origin = [0.0] * len(image_size)
    if image_direction is None:
        image_direction = np.identity(len(image_size)).flatten().tolist()
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
    output_dir: Path,
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
                image_dir = Path(
                    "/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/unicorn"
                    + str(image_input["input_location"])
                )
            else:
                image_dir = Path(image_input["input_location"])

            scan_path = next(image_dir.glob("*.mha"), None)

            if scan_path is None:
                continue
            # neural_representation, _ = encode_path(scan_path)
            # neural_representation["title"] = image_input["interface"]["slug"]
            # outputs.append(neural_representation)

            print(f"Reading image from {scan_path}")
            image = sitk.ReadImage(str(scan_path))
            image = sitk.DICOMOrient(image, desiredCoordinateOrientation="LPS")
            embeddings = encode(image, verbose=True, compression_factor=20) # 128 features
            image_level_neural_representation = {
                "title": image_input["interface"]["slug"],
                "features": embeddings,
            }
            image_representations.append(image_level_neural_representation)
            continue
            
            # Resample image to target spacing
            target_size = [64, 64, 16]
            target_size = [64, 64, 64]
            target_spacing = get_target_spacing(image, target_size=target_size)
            target_spacing[2] /= 3

            neural_representation = extract_features_segmentation(
                image=image,
                model_dir=model_dir,
                domain=domain,
                title=image_input["interface"]["slug"],
                patch_size = target_size,
                patch_spacing=target_spacing,
                overlap_fraction=(0.0, 0.0, 0.5),
            )
            neural_representations.append(neural_representation)

            # merge patchwise feature by averaging and maximizing
            all_features = [p["features"] for p in neural_representation["patches"]]
            all_features = np.array(all_features)
            if all_features.shape[1] != 320:
                raise ValueError(f"Each feature vector must have length 320. Found {all_features.shape[1]}.")
            mean_part = all_features.mean(axis=0)
            max_part = all_features.max(axis=0)
            merged_features = np.concatenate([mean_part, max_part]).tolist()
            if len(merged_features) != 640:
                raise ValueError(f"Merged feature vector should have length 640, but has length {len(merged_features)}.")
            image_level_neural_representation = {
                "title": image_input["interface"]["slug"],
                "features": merged_features,
            }
            image_representations.append(image_level_neural_representation)



        output_path = output_dir / "patch-neural-representation.json"
        write_json_file(location=output_path, content=neural_representations)

        output_path = output_dir / "image-neural-representation.json"
        write_json_file(location=output_path, content=image_representations)


    else:
        print(f"Unknown task: '{task_type}'.")
