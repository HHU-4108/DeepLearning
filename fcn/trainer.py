import fcn_8s
import VOCDataLoder
import torch
from distutils.version import LooseVersion
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader

is_cuda = torch.cuda.is_available()

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        fcn_8s.FCN_8s
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


def load_pretrained_net():
    vgg16_model = models.vgg16(pretrained=True)
    my_model = vgg16_model.features
    return vgg16_model


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        print("----->")
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    print("log_p shape%d" % c)
    print(log_p.shape)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    print(log_p.shape)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def train(path_root):
    # load data
    train_dataloader = VOCDataLoder.VOCLoader(path_root)
    train_data = DataLoader(train_dataloader, batch_size=1, shuffle=True)
    # print(len(train_data))
    # model
    vgg16 = load_pretrained_net()
    model = fcn_8s.FCN_8s(num_calss=21)
    model.copy_weight(vgg16)

    EPOCH = 1
    optim = torch.optim.SGD([
        {'params':get_parameters(model, bias=False)},
        {'params': get_parameters(model, bias=True), 'lr':0.001, 'weight_decay':0},
        ],
        lr=1.0e-4,
        momentum=0.99,
        weight_decay=0.0005)
    if is_cuda:
        model = model.cuda()

    for idx in range(EPOCH):
        for batch_idx, data in enumerate(train_data):
            img, target = data
            # print("train------------>img shape:")
            # print(img.shape)
            if is_cuda:
                img, target = img.cuda().float(), target.cuda()
                img, target = Variable(img), Variable(target)
            optim.zero_grad()
            output = model(img)
            loss = cross_entropy2d(output, target, size_average=False)

            loss /= len(img)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            optim.step()
            metrics = []
            lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(
                    lbl_true, lbl_pred, n_class=21)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)
            log = 'itea:' + [idx] + [loss_data] + [''] * 5 + \
                  metrics.tolist()
            print(log)
    return model


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__ == "__main__":

    train('/home/norman/Dataset/VOCdevkit/VOC2012')


