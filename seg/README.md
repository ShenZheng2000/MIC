# Training

We use a script to automatically
generate and train the configs:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py). The generated configs will be stored
in `configs/generated/`.

# Specific Configs and Checkpoints

<details>
  <summary>Click Here</summary>

TODO: wait till camera-ready to upload these results. 

## Cityscapes -> DarkZurich

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| MIC (HRDA)                | 83 | TODO |
| MIC (HRDA) + Ours         | 84 | TODO |

## Cityscapes -> ACDC

| Experiments | Id | Checkpoints |
|----------|----------|----------|
| MIC (DAFormer)                    | 91 | TODO |
| MIC (DAFormer) + Ours             | 92 | TODO |
| MIC (HRDA)                        | 93 | TODO |
| MIC (HRDA) + Ours                 | 94 | TODO |
| HRDA (DAFormer)                   | 95 | TODO |
| HRDA (DAFormer)  + Ours           | 96 | TODO |
| HRDA (w/o HR-crop)                | 90 | TODO |
| HRDA (w/o HR-crop) + Ours         | 98 | TODO |

</details>