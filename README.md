# MRSegEncoder
>MRI & CT trained feature extraction for the UNICORN challenge.

[MRSegmentator](https://github.com/hhaentze/MRSegmentator) is a segmentation Model for MRI and CT. Given the amount and quality of the training data we can assume that the model knows a lot about human anatomy. So why not use it as a feature extractor?

This repository contains our submission to the [UNICORN](https://unicorn.grand-challenge.org/unicorn/) challenge, a MICCAI 2025 lighthouse challenge for evaluating foundation models (FM). 


## Installation
```bash

# core model
pip install mrsegmentator

# UNICORN baseline framework (only required if you want to run it in a UNICRON setting)
git clone https://github.com/DIAGNijmegen/unicorn_baseline.git
cd unicorn_baseline
pip install .
cd ..

# MRSegEncoder
git clone https://github.com/hhaentze/MRSegEncoder.git
cd MRSegEncoder
pip install -e .


# download smaller weights and update weights path
wget https://github.com/hhaentze/MRSegEncoder/releases/download/v4.4.0/MRSegEncoder_weights_v4.4.zip
unzip -d weights MRSegEncoder_weights_v4.4.zip
export MRSEG_WEIGHTS_PATH="<abs_path_to_weights>"
```

## Inference
#### UNICORN
Make sure to update source and target directories in the code (hardcoded), then run `mrseg_encoder_unicorn`

#### Stand alone
Currently only the python API is usable as a stand alone tool. It receives a 3D image as input and generates a vector of adjustable length (configurable via the "compression_factor" variable).
```python
import SimpleITK as sitk
from mrseg_encoder.inference import encode 

img = sitk.ReadImage(path)
embedding = encode(img,verbose=True)
```