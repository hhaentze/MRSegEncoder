from queue import Queue
from threading import Thread
from typing import Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from tqdm import tqdm
import sys


class Encoder(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def predict_sliding_window_return_logits(
        self, input_image: torch.Tensor
    ) -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            assert (
                input_image.ndim == 4
            ), "input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)"

            if self.verbose:
                print(f"Input shape: {input_image.shape}")
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(
                input_image,
                self.configuration_manager.patch_size,
                "constant",
                {"value": 0},
                True,
                None,
            )

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            predicted_logits = self._internal_predict_sliding_window_return_logits(
                data, slicers, False
            )

            empty_cache(self.device)

        return predicted_logits

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device("cpu")

        def producer(d, slh, q):
            for s in slh:
                q.put(
                    (
                        torch.clone(d[s][None], memory_format=torch.contiguous_format).to(
                            self.device
                        ),
                        s,
                    )
                )
            q.put("end")

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f"move image to device {results_device}")
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # preallocate arrays
            if self.verbose:
                print(f"preallocating results arrays on device {results_device}")
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]),
                dtype=torch.half,
                device=results_device,
            )
            predicted_list = []
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if not self.allow_tqdm and self.verbose:
                print(f"running prediction: {len(slicers)} steps")

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm,file=sys.stdout) as pbar:
                while True:
                    item = queue.get()
                    if item == "end":
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(
                        results_device
                    )
                    predicted_list += [prediction]
                    queue.task_done()
                    pbar.update()
            queue.join()

        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e

        return torch.stack(predicted_list)

    def predict_single_npy_array(
        self,
        input_image: np.ndarray,
        image_properties: dict,
        segmentation_previous_stage: np.ndarray = None,
        output_file_truncated: str = None,
        save_or_return_probabilities: bool = False,
    ):
        """
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy(
            [input_image],
            [segmentation_previous_stage],
            [image_properties],
            [output_file_truncated],
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_threads_in_multithreaded=1,
            verbose=self.verbose,
        )
        if self.verbose:
            print("preprocessing")
        dct = next(ppa)

        if self.verbose:
            print("predicting")

        # Calculate slicers
        data, slicer_revert_padding = pad_nd_image(
            dct["data"], self.configuration_manager.patch_size, "constant", {"value": 0}, True, None
        )
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
        patch_coordinates = []

        for sl in slicers:
            patch_coordinates += [(sl[1].start, sl[2].start, sl[3].start)]

        predicted_logits = self.predict_logits_from_preprocessed_data(dct["data"]).cpu()

        return predicted_logits, patch_coordinates
