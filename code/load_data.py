import tensorflow_datasets as tfds
import pandas as pd


raw_data = tfds.load("movielens/100k-ratings", split="train")
train_ds_raw, test_ds_raw = tfds.load("movielens/100k-ratings", split=['train[:80%]', 'train[80%:]'])

def tfds_to_dataframe(tf_dataset):

    data = [example for example in tf_dataset]

    data = [{key: value.numpy() for key, value in example.items()} for example in data]

    return pd.DataFrame(data)

raw_data_df = tfds_to_dataframe(raw_data)
# train_df = tfds_to_dataframe(train_ds_raw)
# test_df = tfds_to_dataframe(test_ds_raw)


raw_data_df.to_csv('data/movielens.csv', index=False)
print('DataFrame is Saved in `data` folder')
