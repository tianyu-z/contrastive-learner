import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet50
from kornia import augmentation as augs
from kornia import filters

# helper functions


def identity(x):
    return x


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def safe_concat(arr, el, dim=0):
    if arr is None:
        return el
    return torch.cat((arr, el), dim=dim)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# losses


def contrastive_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device
    logits = queries @ keys.t()
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits /= temperature
    return F.cross_entropy(logits, torch.arange(b, device=device))


def nt_xent_loss(queries, keys, temperature=0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(
        ((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)),
        dim=0,
    )
    loss = F.cross_entropy(logits, labels, reduction="sum")
    loss /= n
    return loss


# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):  
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# hidden layer extractor class


class OutputHiddenLayer(nn.Module):
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self._register_hook()

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _register_hook(self):
        def hook(_, __, output):
            self.hidden = output

        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(hook)

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden


# main class


class ContrastiveLearner(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        project_hidden=True,
        project_dim=128,
        augment_both=True,
        use_nt_xent_loss=False,
        augment_fn=None,
        use_bilinear=False,
        use_momentum=False,
        momentum_value=0.999,
        key_encoder=None,
        temperature=0.1,
        batch_size=128,
    ):
        super().__init__()
        self.net = OutputHiddenLayer(net, layer=hidden_layer)

        DEFAULT_AUG = nn.Sequential(
            # RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            # augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            augs.RandomVerticalFlip(),
            augs.RandomSolarize(),
            augs.RandomPosterize(),
            augs.RandomSharpness(),
            augs.RandomEqualize(),
            augs.RandomRotation(degrees=8.0),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((image_size, image_size), p=0.1),
        )
        self.b = batch_size
        self.h = image_size
        self.w = image_size
        self.augment = default(augment_fn, DEFAULT_AUG)

        self.augment_both = augment_both

        self.temperature = temperature
        self.use_nt_xent_loss = use_nt_xent_loss

        self.project_hidden = project_hidden
        self.projection = None
        self.project_dim = project_dim

        self.use_bilinear = use_bilinear
        self.bilinear_w = None

        self.use_momentum = use_momentum
        self.ema_updater = EMA(momentum_value)
        self.key_encoder = key_encoder

        # for accumulating queries and keys across calls
        self.queries = None
        self.keys = None
        random_data = (
            (
                torch.randn(1, 3, image_size, image_size),
                torch.randn(1, 3, image_size, image_size),
                torch.randn(1, 3, image_size, image_size),
            ),
            torch.tensor([1]),
        )
        # send a mock image tensor to instantiate parameters
        self.forward(random_data)

    @singleton("key_encoder")
    def _get_key_encoder(self):
        key_encoder = copy.deepcopy(self.net)
        key_encoder._register_hook()
        return key_encoder

    @singleton("bilinear_w")
    def _get_bilinear(self, hidden):
        _, dim = hidden.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return nn.Parameter(torch.eye(dim, device=device)).to(hidden)

    @singleton("projection")
    def _get_projection_fn(self, hidden):
        _, dim = hidden.shape

        return nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, self.project_dim, bias=False),
        ).to(hidden)

    def reset_moving_average(self):
        assert self.use_momentum, "must be using momentum method for key encoder"
        del self.key_encoder
        self.key_encoder = None

    def update_moving_average(self):
        assert self.key_encoder is not None, "key encoder has not been created yet"
        self.key_encoder = update_moving_average(
            self.ema_updater, self.key_encoder, self.net
        )

    def calculate_loss(self):
        assert (
            self.queries is not None and self.keys is not None
        ), "no queries or keys accumulated"
        loss_fn = nt_xent_loss if self.use_nt_xent_loss else contrastive_loss
        loss = loss_fn(self.queries, self.keys, temperature=self.temperature)
        self.queries = self.keys = None
        return loss

    def adapt_transform_fn(self, x):
        data, frameidx = x
        I0, IFrame, I1 = data
        z = torch.cat((I0, I1), dim=1)
        z_augs = self.augment(z)
        return (z_augs[:, :3], IFrame, z_augs[:, 3:]), frameidx

    def forward(self, x, accumulate=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # b, c, h, w = self.b, 2, self.h, self.w
        # transform_fn = self.augment

        query_encoder = self.net
        # queries = query_encoder(transform_fn(x))
        queries = query_encoder(self.adapt_transform_fn(x))

        key_encoder = (
            self.net
            if not self.use_momentum
            else self._get_key_encoder()
        )
        keys = key_encoder(self.adapt_transform_fn(x))

        if self.use_momentum:
            keys = keys.detach()

        queries, keys = map(flatten, (queries, keys))

        if self.use_bilinear:
            W = self._get_bilinear(keys)
            keys = (W @ keys.t()).t()

        project_fn = (
            self._get_projection_fn(queries) if self.project_hidden else identity
        )
        queries, keys = map(project_fn, (queries, keys))

        self.queries = safe_concat(self.queries, queries)
        self.keys = safe_concat(self.keys, keys)

        return self.calculate_loss() if not accumulate else None
