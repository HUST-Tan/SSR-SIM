def define_Dataset(opt, phase):

    from SSR_data.dataset_2d import Dataset_2d
    dataset = Dataset_2d(opt, phase)

    return dataset