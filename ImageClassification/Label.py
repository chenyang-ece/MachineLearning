
import numpy as np

data = []
labels = []
letters = ['a','b','c','d','e']
label = -1
for letter in letters:
    label += 1
    for num in range(0,20):
        data_path ='picture/' + letter + str(num) +'.jpg' +'.npy'
        data.append( np.load(data_path) )
        labels.append(label)
np.save('data.npy',data)
np.save('labels.npy',labels)
a=np.load('data.npy')
b=np.load('labels.npy')
print(b[20])