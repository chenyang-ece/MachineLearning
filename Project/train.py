import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch.utils.data.dataset as Dataset
import numpy as np
from sklearn import model_selection as ms

#split the dataset
Images = np.load('Images_32.npy')
Labels = np.load('Labels_32.npy')
Images_train, Images_test, Labels_train, Labels_test = ms.train_test_split(Images,Labels,test_size = 0.2)
Images_validation, Images_test, Labels_validation, Labels_test = ms.train_test_split(Images_test,Labels_test,test_size = 0.5)

#save the split dataset
np.save('Images_train.npy',Images_train)
np.save('Images_validation.npy',Images_validation)
np.save('Images_test.npy',Images_test)
np.save('Labels_train.npy',Labels_train)
np.save('Labels_validation.npy',Labels_validation)
np.save('Labels_test.npy',Labels_test)

#setup transform
data_transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize([224, 224]), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#print out which device will be used. GPU is prefered
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

#turn Images_train and Images_validation into uint8
Images_train = np.uint8(Images_train)
Images_validation = np.uint8(Images_validation)

#transform to data, including normalization (-1,1) and changing to tensor data
tensor_Images_train = torch.stack([data_transform(Images_train[i]) for i in range(len(Images_train))])
tensor_Labels_train = torch.stack([torch.Tensor(np.array([Labels_train[i]])) for i in range(len(Labels_train))])
tensor_Images_validation = torch.stack([data_transform(Images_validation[i]) for i in range(len(Images_validation))])
tensor_Labels_validation = torch.stack([torch.Tensor(np.array([Labels_validation[i]])) for i in range(len(Labels_validation))])

#create datset
batch_size = 32
train_dataset = Dataset.TensorDataset(tensor_Images_train,tensor_Labels_train)
validation_dataset = Dataset.TensorDataset(tensor_Images_validation,tensor_Labels_validation)
val_num = len(validation_dataset)

#create dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
validatation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

### --------------------------------------- ###
### Model Configuration ###
## Setting 1: Transfer Learning
net = models.resnet50()
model_weight_path = "./resnet50-pre.pth"
net.load_state_dict(torch.load(model_weight_path), strict=False)
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 5)
net.to(device)
## Setting 2: Learning Rate
LR = 0.01
loss_function = nn.CrossEntropyLoss()
## Setting 3: SGD
optimizer = optim.SGD(net.parameters(), lr=LR, momentum = 0.9)
## Setting 4: output file:
save_path = './SGD'+str(int(1/LR))+'.pth'
# ### --------------------------------------- ###
#
# ### Main ###
best_acc = 0.0
epoch_time = 30
for epoch in range(epoch_time):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        labels = labels.long()
        labels = labels.view(len(labels))
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validatation_loader:
            val_images, val_labels = val_data
            val_labels = val_labels.long()
            val_labels = val_labels.view(len(val_labels))
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f' %
                (epoch + 1, running_loss / step, val_accurate))
print('Finished Training')