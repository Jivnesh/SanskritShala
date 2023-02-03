import numpy as np
from torch import stack

def rearranging_splits(datasets, num_training_samples):
    new_datasets = {}
    data_splits = datasets.keys()
    for split in data_splits:
        if split == 'test' or split == 'poetry' or split == 'prose':
            new_datasets['test'] = datasets['test']
            new_datasets['poetry'] = datasets['poetry']
            new_datasets['prose'] = datasets['prose']
        else:
            num_buckets = len(datasets[split][1])
            num_tensors = len(datasets[split][0][0])
            num_samples = sum(datasets[split][1])
            if num_samples < num_training_samples:
                print("set_num_training_samples (%d) should be smaller than the actual %s size (%d)"
                                 % (num_training_samples, split, num_samples))
            new_datasets[split] = [[[[] for _ in range(num_tensors)] for _ in range(num_buckets)], []]
            new_datasets['extra_' + split] = [[[[] for _ in range(num_tensors)] for _ in range(num_buckets)], []]
    for split in data_splits:
        if split == 'test' or split == 'poetry' or split == 'prose':
            continue
        else:
            curr_bucket_sizes = datasets[split][1]
            curr_samples = datasets[split][0]
            num_tensors = len(datasets[split][0][0])
            curr_num_samples = sum(curr_bucket_sizes)
            sample_indices_in_buckets = {}
            i = 0
            for bucket_idx, bucket_size in enumerate(curr_bucket_sizes):
                for sample_idx in range(bucket_size):
                    sample_indices_in_buckets[i] = (bucket_idx, sample_idx)
                    i += 1
            rng = np.arange(curr_num_samples)
            rng = np.random.permutation(rng)
            sample_indices = {}
            sample_indices[split] = [sample_indices_in_buckets[key] for key in rng[:num_training_samples]]
            sample_indices['extra_' + split] = [sample_indices_in_buckets[key] for key in rng[num_training_samples:]]
            if len(sample_indices['extra_' + split]) == 0:
                if len(sample_indices[split]) > 1:
                    sample_indices['extra_' + split].append(sample_indices[split].pop(-1))
                else:
                    sample_indices['extra_' + split].append(sample_indices[split][0])

            for key, indices in sample_indices.items():
                for bucket_idx, sample_idx in indices:
                    curr_bucket = curr_samples[bucket_idx]
                    for tensor_idx, tensor in enumerate(curr_bucket):
                        new_datasets[key][0][bucket_idx][tensor_idx].append(tensor[sample_idx])
    del datasets
    new_splits = []
    new_splits += [split for split in data_splits if split != 'test' and  split != 'poetry' and  split != 'prose']
    new_splits += ['extra_' + split for split in data_splits if split != 'test' and split != 'poetry' and  split != 'prose']

    for split in new_splits:
        for bucket_idx in range(num_buckets):
            for tensor_idx in range(num_tensors):
                if len(new_datasets[split][0][bucket_idx][tensor_idx]) > 0:
                    new_datasets[split][0][bucket_idx][tensor_idx] = stack(new_datasets[split][0][bucket_idx][tensor_idx])
                else:
                    new_datasets[split][0][bucket_idx] = (1,1)
                    break
            # set lengths of buckets
            if new_datasets[split][0][bucket_idx] == (1,1):
                new_datasets[split][1].append(0)
            else:
                new_datasets[split][1].append(len(new_datasets[split][0][bucket_idx][tensor_idx]))
    return new_datasets