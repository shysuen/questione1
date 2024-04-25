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
