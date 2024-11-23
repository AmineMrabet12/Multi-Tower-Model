import tensorflow_datasets as tfds
import pandas as pd
from tqdm import tqdm


raw_data = tfds.load("movielens/100k-ratings", split="train")
# train_ds_raw, test_ds_raw = tfds.load("movielens/100k-ratings", split=['train[:80%]', 'train[80%:]'])


def tfds_to_dataframe(tf_dataset):

    data = [example for example in tqdm(tf_dataset, desc="Listing Data")]

    data = [{key: value.numpy() for key, value in example.items()} for example in tqdm(data, desc="Dicting Data")]

    return pd.DataFrame(data)


raw_data_df = tfds_to_dataframe(raw_data)
# train_df = tfds_to_dataframe(train_ds_raw)
# test_df = tfds_to_dataframe(test_ds_raw)


raw_data_df.to_csv('data/movielens-100k.csv', index=False)
print('DataFrame is Saved in `data` folder')
