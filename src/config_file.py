config = {
    'data_dir': '../input/',
    'n_splits': 5,
    'train_bs': 16,
    'val_bs': 16,
    'epochs': 64,
    'device': 'cuda:0',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'data_train': 'x_train_224.npy',
    'data_test': 'x_test_224.npy',
    'fold': 0

}
