# Holmberg II pipeline

This folder contains outputs and configs for **Holmberg II**.

## Steps

1) PV slices
python -m src.pv.make_pv --config projects/holmberg_ii/cfg/pv.yaml

Copy code

2) Labels
python -m src.pv.label_pv --config projects/holmberg_ii/cfg/label.yaml

mathematica
Copy code

3) Dataset
python -m src.data.build_dataset --pv_root projects/holmberg_ii/pv --label_root projects/holmberg_ii/labels --out_root projects/holmberg_ii/dataset --patch_pos 512 --patch_vel 96 --vel_channels 128 --pos_fraction 0.5 --splits train=0.7,val=0.2,test=0.1

Copy code

4) Train
python -m src.train.train --dataset projects/holmberg_ii/dataset --out_root projects/holmberg_ii/runs --config configs/train_defaults.yaml

bash
Copy code
