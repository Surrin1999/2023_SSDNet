import os
from sklearn.model_selection import KFold

root = '../../data/KITTI/'


def save_txt(file_names, choose_name, save_path):
    save_names = []
    for i in choose_name:
        save_names.append(file_names[i])
    with open(save_path, 'w') as f:
        for save_name in save_names:
            f.write(save_name + '\n')


def split_dataset(root_path):
    file_list = os.listdir(root_path + 'left/')[:150]
    kf = KFold(n_splits=10, shuffle=True)
    for train, valid in kf.split(file_list):
        save_txt(file_list, train, 'train.txt')
        save_txt(file_list, valid, 'valid.txt')
        break


def get_train_valid():
    train = []
    valid = []
    with open('dataset/train.txt') as f:
        for line in f.readlines():
            train.append(line.rstrip('\n'))
    # with open('dataset/valid.txt') as f:
    #     for line in f.readlines():
    #         valid.append(line.rstrip('\n'))
    return train, valid


def get_test():
    test = []
    with open('dataset/test.txt') as f:
        for line in f.readlines():
            test.append(line.rstrip('\n'))
    return test


def get_test_dataset():
    with open('test.txt', 'w') as f:
        for i in range(150, 200):
            num = "%06d" % i
            f.write(num + '_10.png\n')


if __name__ == '__main__':
    get_test_dataset()
