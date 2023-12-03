# MIC for Domain-Adaptive Semantic Segmentation

## Training

<!-- For convenience, we provide an [annotated config file](configs/mic/gtaHR2csHR_mic_hrda.py)
of the final MIC(HRDA) on GTA→Cityscapes. A training job can be launched using:

```shell
python run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda.py
```

The logs and checkpoints are stored in `work_dirs/`.

For the other experiments in our paper,  -->

We use a script to automatically
generate and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

<!-- 
## Checkpoints

Below, we provide checkpoints of MIC(HRDA) for the different benchmarks.
As the results in the paper are provided as the mean over three random
seeds, we provide the checkpoint with the median validation performance here.

* [MIC(HRDA) for GTA→Cityscapes](https://drive.google.com/file/d/1p_Ytxmj8EckYsq6SdZNZJNC3sgxVRn2d/view?usp=sharing)
* [MIC(HRDA) for Synthia→Cityscapes](https://drive.google.com/file/d/1-Ed0Z2APrhIdsuQTOWXNlZwJJ9Yr2-Vu/view?usp=sharing)
* [MIC(HRDA) for Cityscapes→ACDC](https://drive.google.com/file/d/10RNOAyUY5nYKzIIbNTie458r9etzfvtc/view?usp=share_link)
* [MIC(HRDA) for Cityscapes→DarkZurich](https://drive.google.com/file/d/1HXIwLULUsspBG4U1UAd7OQnDq1G33aTA/view?usp=sharing)

The checkpoints come with the training logs. Please note that:

* The logs provide the mIoU for 19 classes. For Synthia→Cityscapes, it is
  necessary to convert the mIoU to the 16 valid classes. Please, read the
  section above for converting the mIoU.
* The logs provide the mIoU on the validation set. For Cityscapes→ACDC and
  Cityscapes→DarkZurich the results reported in the paper are calculated on the
  test split. For DarkZurich, the performance significantly differs between
  validation and test split. Please, read the section above on how to obtain
  the test mIoU.

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for MIC are:

* [configs/mic/gtaHR2csHR_mic_hrda.py](configs/mic/gtaHR2csHR_mic_hrda.py):
  Annotated config file for MIC(HRDA) on GTA→Cityscapes.
* [experiments.py](experiments.py):
  Definition of the experiment configurations in the paper.
* [mmseg/models/uda/masking_consistency_module.py](mmseg/models/uda/masking_consistency_module.py):
  Implementation of MIC.
* [mmseg/models/utils/masking_transforms.py](mmseg/models/utils/masking_transforms.py):
  Implementation of the image patch masking.
* [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py):
  Implementation of the DAFormer/HRDA self-training with integrated MaskingConsistencyModule

## Acknowledgements

MIC is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS) -->
