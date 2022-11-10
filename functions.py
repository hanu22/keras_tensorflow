#%%
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_skipped = 0
#%%
folder_path = Path("PetImages")
#%%
for fname in folder_path.rglob("*.jpg"):
    fpath = Path(fname)
    print(fpath)

    with fpath.open("rb") as fobj:
        is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)
#%%
