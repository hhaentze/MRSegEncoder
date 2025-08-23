import os
import torch
import torchvision

torchvision.disable_beta_transforms_warning()

from pathlib import Path

from mrseg_encoder.unicorn import vision_new
from unicorn_baseline.io import load_inputs, load_task_description

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")

if vision_new.DEBUG:
    basepath = "/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/unicorn"
    INPUT_PATH = Path(basepath + "/input")
    OUTPUT_PATH = Path(basepath + "/output")
    MODEL_PATH = Path(basepath + "/model")


def print_directory_contents(path: Path | str):
    path = Path(path)
    for child in path.iterdir():
        if child.is_dir():
            print_directory_contents(child)
        else:
            print(child)


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"- number of devices: {torch.cuda.device_count()}")
        print(f"- current device: { (current_device := torch.cuda.current_device())}")
        print(f"- properties: {torch.cuda.get_device_properties(current_device).name}")
    print("=+=" * 10)


def run_vision_and_visionlanguage(input_dir: Path, model_dir: Path, output_dir: Path) -> int:
    """
    Process input data
    """

    task_description = load_task_description(input_path=input_dir / "unicorn-task-description.json")
    input_information = load_inputs(input_path=input_dir / "inputs.json")

    # retrieve task details
    domain = task_description["domain"]
    modality = task_description["modality"]
    task_type = task_description["task_type"]

    if modality == "vision":

        vision_new.run(
            task_description=task_description,
            input_information=input_information,
            model_dir=model_dir,
            output_dir=output_dir,
        )

    elif modality == "vision-language":
        raise ValueError(
            f"Modality '{modality}' and domain '{domain}' not supported by this submission"
        )
    else:
        raise ValueError(f"Modality '{modality}' and domain '{domain}' not supported yet")

    return 0


def run():
    # show GPU information
    _show_torch_cuda_info()

    # print contents of input folder
    print("input folder contents:")
    print_directory_contents(INPUT_PATH)
    print("=+=" * 10)

    # check if the task is image or text
    if (INPUT_PATH / "nlp-task-configuration.json").exists():
        raise ValueError("Language tasks not supported by this submission")
    else:
        return run_vision_and_visionlanguage(INPUT_PATH, MODEL_PATH, OUTPUT_PATH)


def run_charite():
    global INPUT_PATH, OUTPUT_PATH, MODEL_PATH
    
    task = 2    
    INPUT_PATH = Path(f"/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/unicorn/Task{task}/shots-public")
    OUTPUT_PATH = Path(f"/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/unicorn/Task{task}/embeddings")
    MODEL_PATH = Path("/tmp")
      
    for _dir in os.listdir(INPUT_PATH):
        INPUT_PATH = Path(f"/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/unicorn/Task{task}/shots-public/{_dir}")
        OUTPUT_PATH = Path(f"/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/unicorn/Task{task}/embeddings/{_dir}")
        os.makedirs(OUTPUT_PATH,exist_ok=True)

        vision_new.init(
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            model_path=MODEL_PATH,
        )
        run()

if __name__ == "__main__":

    raise SystemExit(run())
