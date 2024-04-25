## 问题与错误描述

本人遇到一个问题。问题为运行以下命令发生报错

```shell
 LD_PRELOAD=./libcnmalloc_override.so python MLU_minst.py
```

报错内容：

```
(pytorch) root@node1:/home# LD_PRELOAD=./libcnmalloc_override.so python MLU_minst.py
Segmentation fault (core dumped)
```

## docker环境配置方法

下载mlu-pytorch镜像

```shell
wget https://sdk.cambricon.com/static/PyTorch/MLU370_1.13_v1.17.0_X86_ubuntu20.04_python3.10_docker/pytorch-v1.17.0-torch1.13.1-ubuntu20.04-py310.tar.gz
```

使用以下shell代码启动镜像

```shell
export MY_CONTAINER="tryxhy"
num=`docker ps -a|grep "$MY_CONTAINER" | wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -it \
        --privileged \
        --pid=host \
        --net=host \
        --device /dev/cambricon_dev0 \
        --device /dev/cambricon_ctl \
        --device /dev/cambricon_ipcm0 \
        --name $MY_CONTAINER \
        -v /home/xhy/try2hijack/:/home \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310 \
        /bin/bash
else
    docker start $MY_CONTAINER
    docker exec -ti $MY_CONTAINER /bin/bash
fi
```

## C++代码编译方法

C++代码如下

```C++
#include "/torch/neuware_home/include/cn_api.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static cn_uint64_t total_allocated_memory = 0;

CNresult custom_cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes);

extern "C" {
    // 导出symbol
    CNresult cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) __attribute__((visibility("default")));
}

CNresult cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) {
    // 调用自建的custom_cnMalloc函数
    return custom_cnMalloc(pmluAddr, bytes);
}

CNresult custom_cnMalloc(CNaddr *pmluAddr, cn_uint64_t bytes) {
    if (bytes > 1000000 || (bytes + total_allocated_memory) > 1000000) {
        printf("Out of Memory Error: Attempted to allocate more than 1,000,000 bytes of MLU memory.\n");
        return CN_MEMORY_ERROR_OUT_OF_MEMORY;
    }
    CNresult (*original_cnMalloc)(CNaddr *, cn_uint64_t) = reinterpret_cast<CNresult (*)(CNaddr *, cn_uint64_t)>(dlsym(RTLD_NEXT, "cnMalloc"));
    if (original_cnMalloc == nullptr) {
        fprintf(stderr, "Error: Unable to find original cnMalloc function.\n");
        return CN_ERROR_INVALID_VALUE;
    }
    CNresult result = original_cnMalloc(pmluAddr, bytes);
    if (result == CN_SUCCESS) {
        total_allocated_memory += bytes;
    }
    return result;
}

//使用静态初始化来模拟constructor的功能
namespace {
    struct Init {
        Init() {
            // 将cnMalloc的地址分配给custom_cnMalloc的地址
            *(void **)(&cnMalloc) = reinterpret_cast<void *>(custom_cnMalloc);
        }
    };
    Init init; // 静态初始化以确保它在main函数之前运行
}
```

编译方法

```shell
g++ -shared -fPIC -o libcnmalloc_override.so cnmalloc_override.cpp -ldl
```

## python脚本内容

```python
import torch 
import torch_mlu 
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import nn
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_data, optimizer, epoch):
    model = model.train()
    for batch_idx, (img, label) in enumerate(train_data):
        img = img.mlu()
        label = label.mlu()
        optimizer.zero_grad()
        out = model(img)
        loss = F.nll_loss(out, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))

def validate(val_loader, model):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, target in val_loader:
            images = images.mlu()
            target = target.mlu()
            output = model(images)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def main():
    data_tf = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.1307],[0.3081])])

    train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True)
    test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

    train_data = DataLoader(train_set, batch_size=64, shuffle=True)
    test_data = DataLoader(test_set, batch_size=1000, shuffle=False)

    net_orig = Net()
    net = net_orig.mlu()
    optimizer = optim.Adadelta(net.parameters(), 1)

    nums_epoch = 1

    save_model = True

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(nums_epoch):
        train(net, train_data, optimizer, epoch)
        validate(test_data, net)

        scheduler.step()
        if save_model: 
            if epoch == nums_epoch-1:
                checkpoint = {"state_dict":net.state_dict(), "optimizer":optimizer.state_dict(), "epoch": epoch}
                torch.save(checkpoint, 'model.pth')

if __name__ == '__main__':
    main()
```

