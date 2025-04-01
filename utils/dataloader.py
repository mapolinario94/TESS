import numpy as np
import torch
from torchvision import transforms, datasets
import tonic
import logging
from .cifar10_dvs import CIFAR10DVS
from .augmentation import ToPILImage, Resize, Padding, RandomCrop, ToTensor, Normalize, RandomHorizontalFlip
import random
import math
import PIL
import warnings


def _is_numpy_image(img):
    return img.ndim in {2, 3}

def cutout(img, i, j, h, w, v, inplace=False):
    """ Erase the CV Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        CV Image: Cutout image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.copy()

    img[i:i + h, j:j + w, :] = v
    return img


class Cutout(object):
    """Random erase the given CV Image.

    It has been proposed in
    `Improved Regularization of Convolutional Neural Networks with Cutout`.
    `https://arxiv.org/pdf/1708.04552.pdf`


    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        pixel_level (bool): filling one number or not. Default value is False
    """
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.pixel_level = pixel_level
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio):
        if type(img) == np.ndarray:
            img_h, img_w, img_c = img.shape
        else:
            img_h, img_w = img.size
            img_c = len(img.getbands())

        s = random.uniform(*scale)
        # if you img_h != img_w you may need this.
        # r_1 = max(r_1, (img_h*s)/img_w)
        # r_2 = min(r_2, img_h / (img_w*s))
        r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if random.random() < self.p:
            left, top, h, w, ch = self.get_params(img, self.scale, self.ratio)

            if self.pixel_level:
                c = np.random.randint(*self.value, size=(h, w, ch), dtype='uint8')
            else:
                c = random.randint(*self.value)

            if type(img) == np.ndarray:
                return cutout(img, top, left, h, w, c, self.inplace)
            else:
                if self.pixel_level:
                    c = PIL.Image.fromarray(c)
                img.paste(c, (left, top, left + w, top + h))
                return img
        return img

def dataloader(args, dataset='DVSGesture', evaluate=False, distributed=False, batch_size=16, val_batch_size=16, workers=4):
    data_path = args.data_path
    if dataset == 'DVSGesture':
        train_loader, val_loader, trainset_len, testset_len = dataloader_gesture(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 11
        args.n_steps = 20
        args.n_inputs = 2
        args.dt = 75e-3
        args.classif = True
        args.delay_targets = 5
        args.skip_test = False
    elif dataset == "CIFAR10DVS":  # Dim: (2, 34, 34)
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar10dvs(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 10
        args.n_inputs = 2
        args.dt = 10e-3
        args.classif = True
        args.delay_targets = 7
        args.skip_test = False
    elif dataset == 'CIFAR10':
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar10(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 6
        args.n_inputs = 32
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    elif dataset == 'CIFAR100':
        train_loader, val_loader, trainset_len, testset_len = dataloader_cifar100(batch_size, val_batch_size, workers, data_path)
        args.full_train_len = trainset_len
        args.full_test_len = testset_len
        args.n_classes = 10
        args.n_steps = 6
        args.n_inputs = 32
        args.dt = 1e-3
        args.classif = True
        args.delay_targets = 5  # 5
        args.skip_test = False
    else:
        logging.info("ERROR: {0} is not supported".format(dataset))
        raise NameError("{0} is not supported".format(dataset))

    return train_loader, val_loader


def dataloader_gesture(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", reproducibility=False):
    labels = 11
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    trainset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=True)
    testset_ori = tonic.datasets.DVSGesture(save_to=data_path, train=False)

    slicing_time_window = 1575000
    slicer = tonic.slicers.SliceByTime(time_window=slicing_time_window)

    frame_transform = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=75000),
        torch.tensor, transforms.Resize(32)
    ])
    frame_transform_test = tonic.transforms.Compose([  # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size,
                                 time_window=75000),
        torch.tensor,
        transforms.Resize(32, antialias=True)
    ])

    trainset_ori_sl = tonic.SlicedDataset(trainset_ori, slicer=slicer,
                                          metadata_path=data_path + '/metadata/online_dvsg_train',
                                          transform=frame_transform)
    # testset_ori_sl = tonic.SlicedDataset(testset_ori, slicer=slicer,
    #                                      metadata_path=data_path + '/metadata/online_dvsg_test',
    #                                      transform=frame_transform_test)

    print(
        f"Went from {len(trainset_ori)} samples in the original dataset to {len(trainset_ori_sl)} in the sliced version.")
    print(
        f"Went from {len(testset_ori)} samples in the original dataset to {len(testset_ori)} in the sliced version.")

    frame_transform2 = tonic.transforms.Compose([  # tonic.transforms.DropEvent(p=0.1),
        torch.tensor,
        transforms.RandomCrop(32, padding=4)
    ])

    trainset = tonic.CachedDataset(trainset_ori_sl,
                                   cache_path=data_path + '/cache/online_fast_dataloading_train',
                                   transform=frame_transform2)
    # if evaluate:
    testset = tonic.CachedDataset(testset_ori,
                                  cache_path=data_path + '/cache/online_fast_dataloading_test',
                                  transform=frame_transform_test)

    if reproducibility:
        import numpy as np
        import random
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True), worker_init_fn=seed_worker, generator=g, )
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=val_batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True), worker_init_fn=seed_worker, generator=g, )
    else:
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True))
        val_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=val_batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            collate_fn=tonic.collation.PadTensors(batch_first=True))

    return train_loader, val_loader, len(trainset_ori_sl), len(testset_ori)


def str_to_num(x):
    labels_dict = {'cup': 0, 'ibis': 1, 'crocodile': 2, 'wild_cat': 3, 'Leopards': 4, 'watch': 5, 'pagoda': 6, 'soccer_ball': 7, 'accordion': 8, 'sunflower': 9, 'rooster': 10, 'ewer': 11, 'stegosaurus': 12, 'ketch': 13, 'rhino': 14, 'cellphone': 15, 'brontosaurus': 16, 'buddha': 17, 'chandelier': 18, 'crayfish': 19, 'strawberry': 20, 'stapler': 21, 'nautilus': 22, 'stop_sign': 23, 'BACKGROUND_Google': 24, 'lamp': 25, 'platypus': 26, 'gerenuk': 27, 'starfish': 28, 'octopus': 29, 'flamingo_head': 30, 'butterfly': 31, 'revolver': 32, 'umbrella': 33, 'garfield': 34, 'sea_horse': 35, 'yin_yang': 36, 'beaver': 37, 'metronome': 38, 'tick': 39, 'trilobite': 40, 'airplanes': 41, 'hawksbill': 42, 'chair': 43, 'pizza': 44, 'anchor': 45, 'euphonium': 46, 'lotus': 47, 'minaret': 48, 'cannon': 49, 'bonsai': 50, 'windsor_chair': 51, 'wrench': 52, 'headphone': 53, 'Motorbikes': 54, 'scorpion': 55, 'cougar_face': 56, 'crocodile_head': 57, 'mandolin': 58, 'barrel': 59, 'inline_skate': 60, 'ferry': 61, 'laptop': 62, 'bass': 63, 'okapi': 64, 'saxophone': 65, 'hedgehog': 66, 'cougar_body': 67, 'scissors': 68, 'crab': 69, 'dalmatian': 70, 'dolphin': 71, 'mayfly': 72, 'pigeon': 73, 'emu': 74, 'electric_guitar': 75, 'panda': 76, 'helicopter': 77, 'schooner': 78, 'camera': 79, 'ant': 80, 'water_lilly': 81, 'elephant': 82, 'llama': 83, 'car_side': 84, 'binocular': 85, 'ceiling_fan': 86, 'menorah': 87, 'dragonfly': 88, 'brain': 89, 'joshua_tree': 90, 'lobster': 91, 'grand_piano': 92, 'flamingo': 93, 'wheelchair': 94, 'dollar_bill': 95, 'kangaroo': 96, 'gramophone': 97, 'Faces_easy': 98, 'snoopy': 99, 'pyramid': 100}
    return torch.tensor(labels_dict[x])


def dataloader_cifar10(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    import torch.utils.data as data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR10


    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    val_loader = data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, len(trainset), len(testset)


def dataloader_cifar100(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets"):
    import torch.utils.data as data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        Cutout(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100


    trainset = dataloader(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = dataloader(root=data_path, train=False, download=False, transform=transform_test)
    val_loader = data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, len(trainset), len(testset)


def dataloader_cifar10dvs(batch_size=16, val_batch_size=16, workers=4, data_path="~/Datasets", img_size=48):
    transform_train = transforms.Compose([
        ToPILImage(),
        Resize(48),
        Padding(4),
        RandomCrop(size=48, consistent=True),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])

    transform_test = transforms.Compose([
        ToPILImage(),
        Resize(48),
        ToTensor(),
        Normalize((0.2728, 0.1295), (0.2225, 0.1290)),
    ])
    num_classes = 10

    trainset = CIFAR10DVS(data_path, train=True, use_frame=True, frames_num=10, split_by='number',
                          normalization=None, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)

    testset = CIFAR10DVS(data_path, train=False, use_frame=True, frames_num=10, split_by='number',
                         normalization=None, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader, len(trainset), len(testset)
