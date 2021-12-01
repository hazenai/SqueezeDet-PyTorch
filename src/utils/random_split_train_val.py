import os
import numpy as np


def main():
    image_set_dir = '../../data/LPR/image_sets_UAE'
    trainval_file = os.path.join(image_set_dir, 'trainval.txt')
    train_file = os.path.join(image_set_dir, 'train.txt')
    val_file = os.path.join(image_set_dir, 'val.txt')

    idx = []
    with open(trainval_file) as f:
        for line in f:
            idx.append(line.strip())

    idx = np.random.permutation(idx)
    print(len(idx))
    val_idx = sorted(idx[:len(idx)])
    val_idx = sorted(idx[:len(idx) // 1])
    train_idx = sorted(idx[len(idx) // 1:])
    print(len(val_idx))
    print(len(train_idx))
    with open(train_file, 'w') as f:
        for i in train_idx:
            f.write('{}\n'.format(i))

    with open(val_file, 'w') as f:
        for i in val_idx:
            f.write('{}\n'.format(i))

    #print('Training set is saved to ', train_file)
    print('Validation set is saved to ', val_file)


if __name__ == '__main__':
    np.random.seed(42)
    main()
