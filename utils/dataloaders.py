import os
import logging
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr


'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    train_a_dir = os.path.join(data_dir, 'train', 'A')
    val_a_dir = os.path.join(data_dir, 'val', 'A')
    train_data = [i for i in os.listdir(train_a_dir) if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(val_a_dir) if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(os.path.join(data_dir, 'train', 'label', img))
    for img in valid_data:
        val_label_paths.append(os.path.join(data_dir, 'val', 'label', img))


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([os.path.join(data_dir, 'train'), img])
    for img in valid_data:
        val_data_path.append([os.path.join(data_dir, 'val'), img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_a_dir = os.path.join(data_dir, 'test', 'A')
    test_data = [i for i in os.listdir(test_a_dir) if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(os.path.join(data_dir, 'test', 'label', img))

    test_data_path = []
    for img in test_data:
        test_data_path.append([os.path.join(data_dir, 'test'), img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def load_image(path):
    with Image.open(path) as image:
        return image.copy()

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = load_image(os.path.join(dir, 'A', name))
    img2 = load_image(os.path.join(dir, 'B', name))
    label = load_image(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']

def transform_sample(sample, aug):
    sample = {
        'image': (sample['image'][0].copy(), sample['image'][1].copy()),
        'label': sample['label'].copy(),
    }

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False, cache_data=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug
        self.cache_data = cache_data
        self.cached_samples = None

        if self.cache_data:
            logging.info('PRELOADING {} samples into RAM'.format(len(self.full_load)))
            self.cached_samples = []
            for index in range(len(self.full_load)):
                img_path = self.full_load[index]['image']
                label_path = self.full_load[index]['label']
                dir = img_path[0]
                name = img_path[1]
                self.cached_samples.append({
                    'image': (
                        load_image(os.path.join(dir, 'A', name)),
                        load_image(os.path.join(dir, 'B', name)),
                    ),
                    'label': load_image(label_path),
                })
            logging.info('FINISHED preloading {} samples into RAM'.format(len(self.cached_samples)))

    def __getitem__(self, index):
        if self.cached_samples is not None:
            return transform_sample(self.cached_samples[index], self.aug)

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
