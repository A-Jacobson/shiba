from torchvision import transforms


class EndTraining(Exception):
    """
    Raised to break out of shiba training loop
    """
    pass


def imagenet_normalize():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def gram_matrix(img):
    (b, ch, h, w) = img.size()
    features = img.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    return features @ features_t / (ch * h * w)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.value = value
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


class ExponentialAverage:
    def __init__(self, smoothing=0.98):
        self.avg = 0
        self.sum = 0
        self.smoothing = smoothing
        self.count = 0
        self.value = 0

    def update(self, value):
        self.value = value
        self.count += 1
        self.sum = self.smoothing * self.sum + \
                   (1 - self.smoothing) * value
        self.avg = self.sum / (1 - self.smoothing ** self.count)

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_momentum(optimizer, momentum):
    for param_group in optimizer.param_groups:
        param_group['momentum'] = momentum


class DotDict(dict):
    """
    Example:
    m = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        # https://github.com/aparo/pyes/issues/183
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]
