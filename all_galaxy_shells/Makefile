GAL?=ngc_2403

pv:
	python -m src.pv.make_pv --config projects/$(GAL)/cfg/pv.yaml

label:
	python -m src.pv.label_pv --config projects/$(GAL)/cfg/label.yaml

dataset:
	python -m src.data.build_dataset --pv_root projects/$(GAL)/pv \
	  --label_root projects/$(GAL)/labels --out_root projects/$(GAL)/dataset \
	  --patch_pos 512 --patch_vel 96 --vel_channels 128 --pos_fraction 0.5

train:
	python -m src.train.train --dataset projects/$(GAL)/dataset \
	  --out_root projects/$(GAL)/runs --config configs/train_defaults.yaml
