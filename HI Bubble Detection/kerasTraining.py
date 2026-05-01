#!/usr/bin/env python3
# TrainUNet_bigRims_20min.py — emphasize large shells (thick-target + focal-tversky + dilation)

import os, time, signal, math, random
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ==== CONFIG ====
PATCH_DIR = "/Users/radish/Desktop/Tsinghua/HI Bubble Detection/patches_256s192"
INDEX_CSV = os.path.join(PATCH_DIR, "index.csv")
IMG_SIZE  = 128
BATCH     = 16
EPOCHS    = 20
LR        = 1e-3
VAL_FRAC  = 0.2
# Heavier emphasis on rims/positives
EDGE_WEIGHT = 3.0
POS_WEIGHT  = 4.0
BASE_CH     = 10
MAX_TRAIN   = 1500
MAX_VAL     = 250
MAX_STEPS_PER_EPOCH = 120
MAX_VAL_STEPS       = 30
USE_MP = False
SEED   = 42
MODEL_OUT = "models/unet_shell_bigRims.keras"
INTERRUPTED_OUT = "models/unet_shell_bigRims_interrupted.keras"
PRINT_EVERY = 1
# ===============

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
if USE_MP:
    from tensorflow.keras import mixed_precision; mixed_precision.set_global_policy("mixed_float16")

# ----- load subset -----
df = pd.read_csv(INDEX_CSV)
paths = df["path"].tolist(); rng = np.random.default_rng(SEED); rng.shuffle(paths)
n_val = max(1, int(len(paths)*VAL_FRAC))
val_paths   = paths[:n_val][:MAX_VAL]
train_paths = paths[n_val:][:MAX_TRAIN]

def load_pack(paths):
    N=len(paths)
    X=np.empty((N,256,256,1),np.float32); Y=np.empty((N,256,256,1),np.float32); R=np.empty((N,256,256,1),np.float32)
    for i,p in enumerate(paths):
        d=np.load(p); X[i,...,0]=d["image"].astype(np.float32); Y[i,...,0]=d["region"].astype(np.float32); R[i,...,0]=d["rim"].astype(np.float32)
    W = 1.0 + EDGE_WEIGHT*R + POS_WEIGHT*Y
    return X,Y,W

Xtr,Ytr,Wtr = load_pack(train_paths)
Xva=Yva=Wva=None
if len(val_paths)>0: Xva,Yva,Wva = load_pack(val_paths)

# ----- tf.data pipeline -----
AUTOTUNE=tf.data.AUTOTUNE

def resize_batch(x,y,w):
    x=tf.image.resize(x,[IMG_SIZE,IMG_SIZE],'bilinear')
    y=tf.image.resize(y,[IMG_SIZE,IMG_SIZE],'nearest')
    w=tf.image.resize(w,[IMG_SIZE,IMG_SIZE],'nearest')
    return x,y,w

def multiscale_aug(x,y,w):
    # random rescale 0.5–1.0 → resize back to 128
    s=tf.random.uniform([],0.5,1.0)
    h=tf.cast(tf.round(IMG_SIZE*s),tf.int32); w2=h
    x=tf.image.resize(x,[h,w2],'bilinear'); y=tf.image.resize(y,[h,w2],'nearest'); ww=tf.image.resize(w,[h,w2],'nearest')
    x=tf.image.resize(x,[IMG_SIZE,IMG_SIZE],'bilinear'); y=tf.image.resize(y,[IMG_SIZE,IMG_SIZE],'nearest'); ww=tf.image.resize(ww,[IMG_SIZE,IMG_SIZE],'nearest')
    # flips/rot
    k=tf.random.uniform([],0,4,dtype=tf.int32); x=tf.image.rot90(x,k); y=tf.image.rot90(y,k); ww=tf.image.rot90(ww,k)
    x=tf.image.random_flip_left_right(x); y=tf.image.random_flip_left_right(y); ww=tf.image.random_flip_left_right(ww)
    x=tf.image.random_flip_up_down(x);   y=tf.image.random_flip_up_down(y);   ww=tf.image.random_flip_up_down(ww)
    # slight contrast/blur
    x=tf.image.random_contrast(x,0.9,1.1)
    # gaussian blur (depthwise conv)
    k = tf.constant([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]],dtype=tf.float32); k/=tf.reduce_sum(k)
    k = tf.reshape(k,[3,3,1,1])
    def maybe_blur(img):
        return tf.cond(tf.random.uniform([])<0.3, lambda: tf.nn.depthwise_conv2d(img,k,[1,1,1,1],'SAME'), lambda: img)
    x=maybe_blur(x)
    return x,y,ww

train_ds=tf.data.Dataset.from_tensor_slices((Xtr,Ytr,Wtr)).shuffle(len(Xtr),seed=SEED,reshuffle_each_iteration=True)\
    .map(resize_batch, num_parallel_calls=AUTOTUNE)\
    .map(multiscale_aug, num_parallel_calls=AUTOTUNE)\
    .batch(BATCH).prefetch(AUTOTUNE)

have_val = Xva is not None
if have_val:
    val_ds=tf.data.Dataset.from_tensor_slices((Xva,Yva,Wva)).map(resize_batch,num_parallel_calls=AUTOTUNE)\
        .batch(BATCH).prefetch(AUTOTUNE)

steps_per_epoch=min(MAX_STEPS_PER_EPOCH, max(1, math.ceil(len(Xtr)/BATCH)))
val_steps = min(MAX_VAL_STEPS, max(1, math.ceil(len(Xva)/BATCH))) if have_val else None
print(f"[INFO] Train {len(Xtr)} | Val {len(Xva) if have_val else 0} | steps/epoch {steps_per_epoch}")

# ----- model: UNet-lite + dilations -----
def sep_block(x,f,d=1):
    x=layers.SeparableConv2D(f,3,padding="same",dilation_rate=d,use_bias=False)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.SeparableConv2D(f,3,padding="same",dilation_rate=d,use_bias=False)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    return x

def unet_lite(input_shape=(IMG_SIZE,IMG_SIZE,1), base=BASE_CH):
    i=layers.Input(shape=input_shape)
    c1=sep_block(i, base);      p1=layers.MaxPool2D()(c1)
    c2=sep_block(p1, base*2);   p2=layers.MaxPool2D()(c2)
    # bottleneck with dilations to boost receptive field
    b1=sep_block(p2, base*4, d=1)
    b2=sep_block(b1, base*4, d=2)
    u2=layers.UpSampling2D()(b2); u2=layers.Concatenate()([u2,c2]); c3=sep_block(u2, base*2)
    u1=layers.UpSampling2D()(c3); u1=layers.Concatenate()([u1,c1]); c4=sep_block(u1, base)
    o=layers.Conv2D(1,1,activation="sigmoid",dtype="float32")(c4)
    return models.Model(i,o)

# ----- losses / metrics -----
def dice(y_true,y_pred,eps=1e-6):
    y_true=tf.cast(y_true,tf.float32); y_pred=tf.cast(y_pred,tf.float32)
    inter=tf.reduce_sum(y_true*y_pred,[1,2,3]); denom=tf.reduce_sum(y_true+y_pred,[1,2,3])
    return tf.reduce_mean((2.*inter+eps)/(denom+eps))

def focal_tversky(y_true,y_pred,alpha=0.7,beta=0.3,gamma=0.75,eps=1e-6):
    # train on *thickened* target to reduce FN on thin rims
    y_true=tf.cast(y_true,tf.float32); y_pred=tf.cast(y_pred,tf.float32)
    y_thick=tf.nn.max_pool2d(y_true, ksize=3, strides=1, padding="SAME")  # 3x3 dilation
    tp=tf.reduce_sum(y_thick*y_pred,[1,2,3])
    fp=tf.reduce_sum((1-y_thick)*y_pred,[1,2,3])
    fn=tf.reduce_sum(y_thick*(1-y_pred),[1,2,3])
    tv=(tp+eps)/(tp+alpha*fn+beta*fp+eps)
    return tf.reduce_mean(tf.pow(1.-tv, gamma))

def combined_loss(y_true,y_pred):
    bce=tf.keras.losses.binary_crossentropy(tf.cast(y_true,tf.float32),tf.cast(y_pred,tf.float32))
    ft =focal_tversky(y_true,y_pred)
    return 0.5*bce + 0.5*ft

model=unet_lite()
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss=combined_loss,
              metrics=[dice, tf.keras.metrics.AUC(curve='PR',name='pr_auc')])

# ----- callbacks -----
class InterruptSave(callbacks.Callback):
    def __init__(self,path): super().__init__(); self.path=path; signal.signal(signal.SIGINT,self.h)
    def h(self,signum,frame): print("\n[INTERRUPT] stopping…"); self.model.stop_training=True
    def on_train_end(self,logs=None):
        try: self.model.save(self.path); print(f"[INTERRUPT] saved {self.path}")
        except: pass

class BatchProg(callbacks.Callback):
    def __init__(self,total,print_every=1): super().__init__(); self.t=total; self.n=0; self.pe=print_every; self.t0=None
    def on_epoch_begin(self,epoch,logs=None): self.n=0; self.t0=time.time(); print(f"[train] Epoch {epoch+1}")
    def on_train_batch_end(self,batch,logs=None):
        self.n+=1
        if self.n%self.pe==0 or self.n==1:
            dt=time.time()-self.t0; sps=(self.n*BATCH)/max(dt,1e-6); rem=self.t-self.n; eta=(dt/max(self.n,1))*rem
            print(f"  step {self.n:>3}/{self.t} | loss={logs.get('loss'):.4f} dice={logs.get('dice'):.4f} prAUC={logs.get('pr_auc'):.4f} | {sps:.1f} samp/s | ETA {eta/60:.1f}m")

ckpt_metric="val_dice" if have_val else "dice"
cb=[InterruptSave(INTERRUPTED_OUT),
    BatchProg(steps_per_epoch, PRINT_EVERY),
    callbacks.ModelCheckpoint(MODEL_OUT, monitor=ckpt_metric, mode="max", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor=ckpt_metric, mode="max", patience=4, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor=ckpt_metric, mode="max", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    callbacks.CSVLogger("training_bigRims.csv", append=False)
]

# ----- train -----
fit_kwargs=dict(x=train_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=cb, verbose=0)
if have_val: fit_kwargs.update(validation_data=val_ds, validation_steps=val_steps)
model.fit(**fit_kwargs)
print(f"[OK] saved best → {MODEL_OUT}")