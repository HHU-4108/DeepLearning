# Norman pytorch branch

leNet5.py
---------
使用Pytorch实现的LeNet5。

1、数据集是pytorch直接下载的minst手写体数据集，代码是从网上直接拷贝的，发现运行不了，第二层卷积之后和全连接层的参数对接出现错误。原来是源码中认为数据集大小是32X32，而实际是28X28.

2、Pytorch中max_pool2d的stride默认是kernel_size，

源码定义如下
    
```
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    Parameters:	
        kernel_size – the size of the window to take a max over
        stride – the stride of the window. Default value is kernel_size
        padding – implicit zero padding to be added on both sides
        dilation – a parameter that controls the stride of elements in the window
        return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
        ceil_mode – when True, will use ceil instead of floor to compute the output shape
```


Pretrain_AlexNet.py
---------
Use pretrained model of AlexNet.

The compression method has a great effect on the result.

used

```
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
```

then get better result:
```
output label:n02123159 tiger cat tensor([282], device='cuda:0')
```
