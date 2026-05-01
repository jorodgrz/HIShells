import tensorflow as tf
from src.pv.dataset import build_dataset

def test_build_dataset_smoke():
    ds = build_dataset("pv_config.yaml", "train", batch_size=2, seed=1)
    x,y = next(iter(ds))
    assert x.shape[-1]==1 and y.shape[-1]==1