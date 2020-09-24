import numpy as np
from PIL import Image
a=np.load('data.npy')
b=np.load('labels.npy')
print(b)
im = Image.fromarray(a[:])


im.save("your_file.jpeg")