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

import json
import math
import ntpath
from os.path import join
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mrseg_encoder.adapted_nnInference import Encoder
from mrseg_encoder.simpleitk_reader_writer import SimpleITKIO
from mrsegmentator import config

config.disable_nnunet_path_warnings()

def spatial_compress(x, compression_factor=2):
    # compression_factor = 2 results in 1280 features
    # compresision_factor = 8 results in 320 features
    # compresision_factor = 16 results in 160 features

    # x shape: [320, 4, 4, 4]
    if (320 % compression_factor) != 0:
        raise ValueError(f"Input feature dimension {x.shape[0]} must be divisible by {compression_factor}.")
    
    x = torch.squeeze(x)  # Ensure x is a 4D tensor [320, 4, 4, 4]

    # Save original dtype
    orig_dtype = x.dtype

    # Cast to float32 if dtype is half to avoid pooling errors on CPU
    if orig_dtype == torch.float16:
        x = x.to(torch.float32)
    
    # Step 1: Average pooling to compress spatial dimensions
    # From [320, 4, 4, 4] to [320, 2, 2, 2]
    pooled = F.avg_pool3d(x, kernel_size=2, stride=2)  # [320, 2, 2, 2]
    
    # Step 2: Reshape to group features for aggregation
    # We want to process every 2 consecutive features (320/2 = 160 groups)
    pooled = pooled.view(320//compression_factor, compression_factor, 2, 2, 2)  # [80, 4, 2, 2, 2]
    
    # Step 3: Calculate average and maximum across the 4 consecutive features
    avg_features = torch.mean(pooled, dim=1)  # [16, 2, 2, 2]
    
    # Step 4: Transpose to get spatial-first organization
    # We want [2, 2, 2, 160] so each spatial position has 160 features
    combined = avg_features.permute(1, 2, 3, 0)  # [2, 2, 2, 160]
    
    # Step 5: Flatten to final 1D representation
    result = combined.flatten()  # [2*2*2*160] = [8*160] = [1280]
    
    return result


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


def most_centered_tuple_index(points: List[Tuple[int, int, int]]) -> int:
    if not points:
        raise ValueError("The list of points is empty.")

    # Step 1: Compute centroid (still floats)
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)
    z_mean = sum(p[2] for p in points) / len(points)
    centroid = (x_mean, y_mean, z_mean)

    # Step 2: Compute Euclidean distances from each int-coordinate tuple to centroid
    def distance(p: Tuple[int, int, int]) -> float:
        return math.sqrt(
            (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2 + (p[2] - centroid[2]) ** 2
        )

    # Step 3: Find the index of the point with the smallest distance
    distances = [distance(p) for p in points]
    return distances.index(min(distances))


def remove_decoder(model):

    if hasattr(model.network, "decoder"):
        delattr(model.network, "decoder")

    def encoder_forward(self, x):
        # Use only the encoder part
        skips = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        return x, skips  # Return final features + all intermediate features

    model.network.forward = encoder_forward.__get__(model.network, type(model.network))

    decoder_keys = []

    for p in model.list_of_parameters:
        for k in p.keys():
            if "decoder" in k:
                decoder_keys += [k]

    for k in decoder_keys:
        del model.list_of_parameters[0][k]

    return model



def static_encoder():

    # initialize weights directory
    config.setup_mrseg()

    # initialize encoder
    encoder = Encoder(
        tile_step_size=1,
        use_gaussian=False,
        use_mirroring=False,
        device=torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    # initialize the network architecture, load the checkpoints
    encoder.initialize_from_trained_model_folder(
        config.get_weights_dir(),
        use_folds=[0],
        checkpoint_name="checkpoint_final.pth",
    )

    # remove decoder parts
    encoder = remove_decoder(encoder)

    return encoder


def encode(image, verbose: bool = False, compression_factor=2):
# def encode(npy_image, props):
    global encoder

    img, props, itk_image  = SimpleITKIO().transform_image(image,verbose=verbose,force_spacing=True)
    # # initialize weights directory
    # config.setup_mrseg()

    # # initialize encoder
    # encoder = Encoder(
    #     tile_step_size=1,
    #     use_gaussian=False,
    #     use_mirroring=False,
    #     device=torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda", 0),
    #     verbose=False,
    #     verbose_preprocessing=False,
    #     allow_tqdm=False,
    # )

    # # initialize the network architecture, load the checkpoints
    # encoder.initialize_from_trained_model_folder(
    #     config.get_weights_dir(),
    #     use_folds=[0],
    #     checkpoint_name="checkpoint_final.pth",
    # )

    # # remove decoder parts
    # encoder = remove_decoder(encoder)

    # inference
    try:
        embeddings, coordinates = encoder.predict_single_npy_array(img, props)
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return [0]*320

    # reduce embedding size from     to 320]
    small_embeddings = []
    for i in range(len(coordinates)):
        #gap = F.adaptive_avg_pool3d(embeddings[i], 1).view(1, 320)
        # gmp = F.adaptive_max_pool3d(embeddings[i], 1).view(1, 320)
        # x = torch.cat([gap, gmp], dim=1)

        # # 1) Split along depth
        # x1, x2 = (
        #     embeddings[i][:, :, :3, :, :],
        #     embeddings[i][:, :, 3:, :, :],
        # )  # both are [B, 320, 3, 4, 5]

        # # 2) Average-pool each half
        # gap1 = F.adaptive_avg_pool3d(x1, 1).view(1, 320)  # “top” summary
        # gap2 = F.adaptive_avg_pool3d(x2, 1).view(1, 320)  # “bottom” summary

        # # 3) Concatenate
        # pooled = torch.cat([gap1, gap2], dim=1)  # → [1, 640]

        # small_embeddings += [pooled[0].tolist()]
        if verbose: print("DEBUG: embeddings shape:", embeddings[i].shape)
        spatial_embedding = spatial_compress(embeddings[i][0], compression_factor)
        if verbose: print("DEBUG: spatial_embedding shape:", spatial_embedding.shape, type(spatial_embedding))
        small_embeddings += [spatial_embedding.tolist()]

    # return only first slice

    if verbose: print(f"Extracted {len(small_embeddings)} embeddings")
    return small_embeddings[0]



def infer(
    img_path: str,
    outdir: str,
    fold: int = 0,
    verbose: bool = False,
    cpu_only: bool = False,
    return_results: bool = False,
    is_pathology_wsi: bool = False,
) -> None:
    global encoder
    

    # make output directory
    if not return_results:
        Path(outdir).mkdir(exist_ok=True)

    # # initialize encoder
    # encoder = Encoder(
    #     tile_step_size=1,
    #     use_gaussian=False,
    #     use_mirroring=False,
    #     device=torch.device("cpu") if cpu_only else torch.device("cuda", 0),
    #     verbose=verbose,
    #     verbose_preprocessing=verbose,
    #     allow_tqdm=True,
    # )

    # # initialize the network architecture, load the checkpoints
    # encoder.initialize_from_trained_model_folder(
    #     config.get_weights_dir(),
    #     use_folds=[fold],
    #     checkpoint_name="checkpoint_final.pth",
    # )

    # # remove decoder parts
    # encoder = remove_decoder(encoder)

    # load image
    img, props, itk_image = SimpleITKIO().read_image(
        img_path, verbose=True, is_pathology_wsi=is_pathology_wsi
    )

    # inference
    embeddings, coordinates = encoder.predict_single_npy_array(img, props)

    # reduce embedding size from [320, 6, 4, 5] to [640]
    small_embeddings = []
    for i in range(len(coordinates)):
        gap = F.adaptive_avg_pool3d(embeddings[i], 1).view(1, 320)
        gmp = F.adaptive_max_pool3d(embeddings[i], 1).view(1, 320)
        x = torch.cat([gap, gmp], dim=1) # → [1, 640]
        small_embeddings += [x[0].tolist()]
        # # 1) Split along depth
        # x1, x2 = (
        #     embeddings[i][:, :, :3, :, :],
        #     embeddings[i][:, :, 3:, :, :],
        # )  # both are [B, 320, 3, 4, 5]

        # # 2) Average-pool each half
        # gap1 = F.adaptive_avg_pool3d(x1, 1).view(1, 320)  # “top” summary
        # gap2 = F.adaptive_avg_pool3d(x2, 1).view(1, 320)  # “bottom” summary

        # # 3) Concatenate
        # pooled = torch.cat([gap1, gap2], dim=1)  # → [1, 640]

        # small_embeddings += [pooled[0].tolist()]

    # inverse coordinate positions
    # (the current coordinates are for numpy arrays. However, when I transformed the
    # sitk image to an array I implicitely changed the dimensons from x,y,z to z,y,x.
    # Consequently, I need to reverse them again to make it compatible with sitk)
    patch_size = [64,64,64]
    coordinates = [coord for coord in coordinates]

    title = ntpath.basename(img_path)

    # transform coordinates to real world values
    new_coords = []

    for start_coords in coordinates:

        x, y, z = start_coords

        if is_pathology_wsi:
            matrix_coordinates = (
                (x, y),
                (x + patch_size[0], y + patch_size[1]),
            )
        else:
            matrix_coordinates = (
                start_coords,
                (x + patch_size[0], y + patch_size[1], z + patch_size[2]),
            )

        world_coordinates = tuple(
            itk_image.TransformIndexToPhysicalPoint(coord) for coord in matrix_coordinates
        )
        new_coords.append(world_coordinates)

    # create patches
    patches = []
    for i, coord in enumerate(new_coords):
        patches += [
            {
                "features": small_embeddings[i],
                "coordinates": list(coord[0]),
            }
        ]
    # combine with meta
    patch_level_neural_representation = make_patch_level_neural_representation(
        patch_features=patches,
        patch_size=patch_size,
        patch_spacing=[1.0, 1.0, 1.0],
        image_size=itk_image.GetSize(),
        image_origin=itk_image.GetOrigin(),
        image_spacing=itk_image.GetSpacing(),
        image_direction=itk_image.GetDirection(),
        title="patch-level-neural-representation",
    )

    center_id = most_centered_tuple_index(coordinates)
    image_level_neural_representation = {
        "title": title,
        "features": small_embeddings[center_id],
    }
    if return_results:
        return image_level_neural_representation, patch_level_neural_representation

    # save
    with open(join(outdir, "patch-neural-representation.json"), "w") as f:
        json.dump(
            [patch_level_neural_representation],
            f,
            indent=4,
        )

    # save central patch for single patch tasks
    center_id = most_centered_tuple_index(coordinates)
    with open(join(outdir, "image-neural-representation.json"), "w") as f:
        json.dump(
            [
                {
                    "title": title,
                    "features": small_embeddings[center_id],
                }
            ],
            f,
            indent=4,
        )

# initialize the static encoder
print("Initializing static encoder...")
encoder = static_encoder()