import tensorflow as tf
import os

def create_siamese_dataset(datasets, augment_func=None, augment_flags=None, preprocess=False, shuffle_buffer=2**8, rei=True, batch_size=64, seed=0, prefetch=tf.data.AUTOTUNE, valid_no=None, train_process_func=None, valid_process_func=None, num_parallel_calls=tf.data.AUTOTUNE):
    if valid_no is None:
        valid_take_number=0
    else:
        valid_take_number=valid_no
        
    if augment_flags is not None:
        assert len(augment_flags)==len(datasets)
    else:
        augment_flags = [True]*len(datasets)
        
    if train_process_func is not None:
        if isinstance(train_process_func, list):
            train_process_func_list = train_process_func
        elif callable(train_process_func):
            train_process_func_list = [train_process_func]*len(datasets)
        else:
            raise Exception("train_process_list is not callable or list")
            
    if valid_process_func is not None:
        if isinstance(valid_process_func, list):
            valid_process_func_list = valid_process_func
        elif callable(valid_process_func):
            valid_process_func_list = [valid_process_func]*len(datasets)
        else:
            raise Exception("valid_process_list is not callable or list")
            
    if not isinstance(datasets, list):
        raise Exception("datasets must be a list")
        
    if isinstance(augment_func, list):
        augment_func_list = augment_func
    elif callable(augment_func):
        augment_func_list = [augment_func]*len(datasets)
    else:
        raise Exception("augment_func is not callable or list")
    
    train_ds_list = []
    if valid_no is not None:
        valid_ds_list = []
    for i, dataset in enumerate(datasets):
        if valid_no is not None:
            valid_ds = dataset.take(valid_take_number)
            train_ds = dataset.skip(valid_take_number)
        else:
            train_ds = dataset
        
        if train_process_func is not None:
            train_ds = train_ds.map(train_process_func_list[i], num_parallel_calls=num_parallel_calls)
        if valid_process_func is not None:
            valid_ds = valid_ds.map(valid_process_func_list[i], num_parallel_calls=num_parallel_calls)
        if augment_func_list[i] is not None and augment_flags[i]:
            train_ds_list.append(train_ds.map(augment_func_list[i], num_parallel_calls=num_parallel_calls))
        else:
            train_ds_list.append(train_ds)
        if valid_no is not None:
            valid_ds_list.append(valid_ds)
#     train_data = tf.data.Dataset.zip((*train_ds_list,))
#     if valid_no is not None:
#         valid_data = tf.data.Dataset.zip((*valid_ds_list,))
    if preprocess:
        for j, train_data in enumerate(train_ds_list):
            train_ds_list[j] = train_data.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=rei).batch(batch_size, drop_remainder=True).prefetch(prefetch)
    if valid_no is not None:
        return (*train_ds_list, *valid_ds_list,)
    else:
        return (*train_ds_list,)


# def load_dataset(whole_patch_dir, split_patch_dir=None, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
#     whole_patch_list    = [os.path.join(whole_patch_dir, filename) for filename in os.listdir(whole_patch_dir)]
#     wp_dataset          = tf.data.TFRecordDataset(tf.data.Dataset.list_files(whole_patch_list, shuffle=shuffle))
#     wp_dataset          = wp_dataset.map(_parse_record, num_parallel_calls=num_parallel_calls).map(_parse_tensor, num_parallel_calls=num_parallel_calls)
#     if split_patch_dir:
#         split_patch_list = [os.path.join(split_patch_dir, filename) for filename in os.listdir(split_patch_dir)]
#         sp_dataset       = tf.data.TFRecordDataset(tf.data.Dataset.list_files(split_patch_list, shuffle=shuffle))
#         sp_dataset       = sp_dataset.map(_parse_record, num_parallel_calls=num_parallel_calls).map(_parse_tensor, num_parallel_calls=num_parallel_calls)
#         return wp_dataset, sp_dataset
#     return wp_dataset

def load_datasets(ds_path_list, feature_description, output_dict, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
    
    @tf.function
    def _parse_record(x):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(x, feature_description)
    
    
    @tf.function
    def _parse_tensor(x):
        output = {key:tf.io.parse_tensor(x[key],out_type=output_dict[key]) for key in output_dict.keys()}
        return output
    
    if not isinstance(ds_path_list, list):
        ds_dir_list = [ds_path_list]
    else:
        ds_dir_list = ds_path_list
    output_list = []
    for ds_dir in ds_dir_list:
        ds_list = [os.path.join(root, filename) for root,_,files in os.walk(ds_dir) for filename in files]
        dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(ds_list, shuffle=shuffle))
        dataset = dataset.map(_parse_record, num_parallel_calls=num_parallel_calls).map(_parse_tensor, num_parallel_calls=num_parallel_calls)
        output_list.append(dataset)
    return (*output_list,)

def preprocess_ds(ds, batch_size, shuffle_no, seed, pre, rei=True, drop_remainder=True):
    return ds.shuffle(shuffle_no, seed=seed, reshuffle_each_iteration=rei).batch(batch_size, drop_remainder=drop_remainder).prefetch(pre)
