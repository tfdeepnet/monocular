import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm


def train(epoch,  trainloader, optimizer, net, criterion, num_bucket,device):
    net.train()
    running_loss = 0.0
    correct = 0
    processed = 0

    pbar = tqdm(trainloader)

    for idx, data in enumerate(pbar) : #trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        ##print("count {}".format(i))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        processed += inputs.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        # running_loss += loss.item()
        # if idx % (num_bucket) == 0:    # print every 2000 mini-batches
        #      print('[%d, %5d] loss: %.3f' %
        #            (epoch + 1, idx + 1, running_loss / (num_bucket)))
        #      running_loss = 0.0

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={idx} Accuracy={100*correct/processed:0.2f}')

train_losses = []
train_acc = []

def train_metrics(trainloader, net, criterion , device ):
    correct = 0
    total = 0
    train_loss = 0

    net.eval()
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            train_loss += criterion(outputs, labels).item()  # sum up batch loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    trainsetsize = len(trainloader.dataset)
    train_loss /= trainsetsize
    train_losses.append(train_loss)

    train_acc.append(100. * correct / trainsetsize)
    return train_losses,train_acc