Note: Before running train.py and test.py, please download "resnet50-pre.pth" and "SGD100.pth" from https://1drv.ms/u/s!AuvCiXg8SwEPsRp3XArwWVhrykoD?e=tcVTnR

resnet50-pre.th: The trained model provided by pytorch (https://download.pytorch.org/models/resnet50-19c8e357.pth). It's used for transfer learning.
SGD100.pth: The optimized resnet50 model that we trained.


About our code:
train.py: It can load and split dataset into train, validation and test dataset. And then, it will train the resnet50 model with transfer learning and SGD optimizer. (Learning rate is in the line 60, you can modify it) 

test.py: It will load our optimized resnet50 model "SGD100.pth" and generate predict accuracy and confusion matrix. 
Note: the test dataset name have to be "Images_test.npy" and "Labels_test.npy". Or you can change it in the line 20, 21.

