# NGC 2403 pipeline

This folder contains outputs and configs for **NGC 2403**.

## Steps

1) PV slices
python -m src.pv.make_pv --config projects/ngc_2403/cfg/pv.yaml

Copy code

2) Labels
python -m src.pv.label_pv --config projects/ngc_2403/cfg/label.yaml

mathematica
Copy code

3) Dataset
python -m src.data.build_dataset --pv_root projects/ngc_2403/pv --label_root projects/ngc_2403/labels --out_root projects/ngc_2403/dataset --patch_pos 512 --patch_vel 96 --vel_channels 128 --pos_fraction 0.5 --splits train=0.7,val=0.2,test=0.1

Copy code

4) Train
python -m src.train.train --dataset projects/ngc_2403/dataset --out_root projects/ngc_2403/runs --config configs/train_defaults.yaml

bash
Copy code
