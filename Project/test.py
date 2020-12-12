import torch
from torchvision import transforms, models
import torch.utils.data.dataset as Dataset
import numpy as np
from sklearn.metrics import confusion_matrix

#setup transform
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224]), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#print out which device will be used. GPU is prefered
print("Active GPU / CPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

#read test dataset
# If you just trained a model, save as a .pth file and
# then run this test.py immediately, the test dataset
# you are using is exactly all data that haven't been used in training
# if you need a new random test dataset, please Run datasplit.py #
Images_test = np.uint8(np.load('Images_test.npy'))
Labels_test = np.load('Labels_test.npy')
print('Finish loading Data and Label')

print("Data preprocessing")
#transform the data
tensor_Images_test = torch.stack([data_transform(Images_test[i]) for i in range(len(Images_test))])
tensor_Labels_test = torch.stack([torch.Tensor(np.array([Labels_test[i]])) for i in range(len(Labels_test))])

#create dataset
test_dataset = Dataset.TensorDataset(tensor_Images_test,tensor_Labels_test)

#creat dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32)

#load model weights
net = models.resnet50(num_classes=5)
model_weight_path = "./SGD100.pth"
net.load_state_dict(torch.load(model_weight_path, map_location=device))
print('Finish loading Model')

#predict class
print('Start Predicting')
net.eval()
all_preds = torch.tensor([])
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data
        outputs = net(test_images)
        _,preds = torch.max(outputs,1)
        all_preds = torch.cat((all_preds, preds),dim=0)

test_predicted = all_preds.numpy().tolist()
test_true = Labels_test.tolist()

print('The predicted label vector:', test_predicted)
print('The true label vector:', test_true)

c = confusion_matrix(test_true,test_predicted)
predict_accuracy = np.sum(c.diagonal()) / np.sum(c)
print('Accuracy of prediction is ', predict_accuracy)
print('Confusion matrix is as follow:')
print(c)


