#!/bin/env python3

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


def load_cat_v_dog_dataset(val_size=0.2):
    ds, ds_info = tfds.load('cats_vs_dogs', split='train', with_info=True)

    #fig  = tfds.show_examples(test_ds, test_ds_info)
    #fig.show()

    ds_size = len(ds)
    val_ds_size = int(val_size*ds_size)
    
    train_ds = ds.skip(val_ds_size).prefetch(10)
    val_ds = ds.take(val_ds_size).prefetch(10)

    return train_ds, val_ds


def main():
    
    train_ds, val_ds = load_cat_v_dog_dataset()

    print('# Training images: ', len(train_ds))
    print('# Validation images: ', len(val_ds))
    
    print(train_ds)
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break 

if __name__ == '__main__':
    main()
