# NOTE: hardcode v2 now for debugs
run_experiment() {
    GPU_ID="$1"
    EXP_ID="$2"
    
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    nohup python run_experiments.py \
    --exp "$EXP_ID" \
    > "outs/${EXP_ID}_v3.out" 2>&1 &
}

# run_experiment 0 80
# run_experiment 2 88

# run_experiment 0 90
# run_experiment 1 98


# NOTE: use xxx_v2.out for all exps now



# DAFormer + MIC (CS => DZ)
# run_experiment 0 81              # 81 w/o warping
# run_experiment 2 82              # 82 w/ warping

# DAFormer + MIC (CS => ACDC)
# run_experiment 1 91              # 91 w/o warping
# run_experiment 0 92              # 92 w/ warping




# DAFormer + HRDA + MIC (CS => DZ)
# run_experiment 2 83                 # 83: w/o warping
# run_experiment 1 84                  # 84: w/ warping

# DAFomer + HRDA + MIC (CS => ACDC)
# run_experiment 0 93                  # 93: w/o warping
# run_experiment 2 94                   # 94: w/ warping

# # DAFormer + HRDA (CS => DZ) => Skip for now
# run_experiment 1 85                     # 85: w/o warping
# # run_experiment 0 86                     # 86: w/ warping

# DAFormer + HRDA (CS => ACDC)
# run_experiment 1 95                     # 95: w/o warping
# run_experiment 0 96                     # 96: w/ warping
