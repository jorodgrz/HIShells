# src/train/train_pv_unet.py
from __future__ import annotations
from datetime import datetime

# --- keep TF chatty logs down before importing TF ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 3=errors only

import json
from pathlib import Path
import warnings

# astrophysics FITS header warnings can be noisy
from astropy.wcs.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras

# absl sometimes prints INFO; set to ERROR if available
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
tf.get_logger().setLevel("ERROR")
keras.utils.disable_interactive_logging()

# --- project imports ---
from src.utils.config import resolve_config
from src.pv.dataset import build_dataset, estimate_steps
from src.train.models_unet import unet_pv
from src.train.losses import make_loss_and_metrics



class EveryNEpochs(keras.callbacks.Callback):
    """
    Print compact metrics every N epochs (and also at epoch 1 and the last).
    Works best with model.fit(..., verbose=0).
    """
    def __init__(self, every: int = 5):
        super().__init__()
        self.every = max(1, int(every))
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        # Try to capture total epochs from params if Keras provides it
        self.total_epochs = self.params.get("epochs", None)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        e = epoch + 1
        is_first = (e == 1)
        is_multiple = (e % self.every == 0)
        is_last = (self.total_epochs is not None and e == self.total_epochs)
        if is_first or is_multiple or is_last:
            fields = ("loss","pr_auc","precision","recall",
                      "val_loss","val_pr_auc","val_precision","val_recall")
            msg = [f"epoch {e}"]
            for k in fields:
                if k in logs and logs[k] is not None:
                    try:
                        msg.append(f"{k}={float(logs[k]):.4f}")
                    except Exception:
                        # if it's a tensor or something odd
                        msg.append(f"{k}={logs[k]}")
            print("[progress]", ", ".join(msg), flush=True)


def _default_run_name(prefix="pv_unet"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _build_model(cfg: dict) -> keras.Model:
    """
    Construct and compile the U-Net for PV slices using config.
    """
    # model
    m = unet_pv(
        patch_shape=(int(cfg["train"]["patch_vel"]),
                     int(cfg["train"]["patch_pos"]),
                     1),
        base_filters=int(cfg["model"]["base_filters"]),
        depth=int(cfg["model"]["depth"]),
        dilation_rate=int(cfg["model"]["dilation_rate"]),
        dropout=float(cfg["model"]["dropout"]),
    )

    # loss & metrics
    loss, metrics = make_loss_and_metrics(cfg)

    # optimizer
    opt = keras.optimizers.Adam(
        learning_rate=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"])
    )

    m.compile(optimizer=opt, loss=loss, metrics=metrics)
    return m


def train(cfg_path: str,
          run_name: str | None = None,
          quiet: bool = True,
          print_every: int = 5) -> None:
    """
    Main training entrypoint.
    - quiet=True -> Keras verbose=0; we print compact lines via EveryNEpochs.
    - print_every controls how often those compact lines appear.
    """
    # Resolve and echo resolved config (also writes data/_resolved_config.yaml)
    cfg = resolve_config(cfg_path, write_resolved=True)

    # Output dir
    out = Path("runs") / (run_name or _default_run_name())
    out.mkdir(parents=True, exist_ok=True)

    # Batching / epochs
    bs = int(cfg["optim"]["batch_size"])
    epochs = int(cfg["optim"]["epochs"])

    # Datasets (repeat + fixed steps avoid OUT_OF_RANGE spam)
    ds_train = build_dataset(cfg_path, split="train", batch_size=bs, seed=42,   repeat=True)
    ds_val   = build_dataset(cfg_path, split="val",   batch_size=bs, seed=4242, repeat=True)

    steps_per_epoch  = max(1, estimate_steps(cfg_path, "train", bs))
    validation_steps = max(1, estimate_steps(cfg_path, "val",   bs))

    model = _build_model(cfg)

    # Callbacks
    cbs: list[keras.callbacks.Callback] = [
        EveryNEpochs(every=print_every),
        keras.callbacks.ModelCheckpoint(
            filepath=str(out / "best_model.keras"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc", mode="max",
            factor=0.5, patience=6, min_lr=1e-6, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=12, restore_best_weights=True, verbose=0
        ),
        keras.callbacks.CSVLogger(str(out / "history.csv"), append=True),
    ]

    # Verbosity: 0 = silent (our EveryNEpochs prints), 2 = per-epoch single line
    verbose = 0 if quiet else 2

    # Fit
    hist = model.fit(
        ds_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=verbose,
    )

    # Final save (native .keras format)
    model.save(out / "final_model.keras")
    (out / "history_final.json").write_text(json.dumps(hist.history, indent=2))
    print(f"[done] saved best/last to {out}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to pv_config.yaml")
    ap.add_argument("--run", default=None, help="Run subfolder name under runs/")
    ap.add_argument("--quiet", action="store_true", help="Use compact logging via EveryNEpochs")
    ap.add_argument("--every", type=int, default=5, help="Print every N epochs when --quiet")
    args = ap.parse_args()

    train(args.config, run_name=args.run, quiet=args.quiet, print_every=args.every)