import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

with open('data/train/fr_crops.p', 'rb') as f:

    fr_crops = pickle.load(f)

with open('data/train/en_crops.p', 'rb') as f:

    en_crops = pickle.load(f)

fr_labels = np.ones((fr_crops.shape[0],))
en_labels = np.zeros((en_crops.shape[0],))

print(fr_crops.shape, en_crops.shape)
print(fr_labels.shape, en_labels.shape)

crops = np.concatenate((fr_crops, en_crops))
labels = np.concatenate((fr_labels, en_labels))



crops, labels = shuffle_in_unison(crops, labels)

crops = np.expand_dims(crops, -1)

print(crops.shape, labels.shape)

# There appear to be a lot of mislabelled training images...

'''
for i in range(30):

    print(labels[i])

    plt.figure()
    plt.imshow(crops[i])
    plt.show()
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

batch_size = 16


for epoch in range(3):

    for i in range(100):

        batch_indices = range(batch_size * i, batch_size * (i + 1))
        data_batch = torch.Tensor(crops.take(batch_indices, axis = 0, mode = 'wrap'))
        label_batch = torch.Tensor(labels.take(batch_indices, mode = 'wrap'))

        print(data_batch.shape, label_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 10 == 9:
            # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
