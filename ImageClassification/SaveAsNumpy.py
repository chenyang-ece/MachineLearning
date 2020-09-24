
import numpy as np
import imageio
import os
os.chdir('picture')     #切换python工作路径到你要操作的图片文件夹，mri_2d_test为我的图片文件夹
   #利用np.ones()函数生成一个三维数组，当然也可用np.zeros，此数组的每个元素a[i]保存一张图片
i=0
for filename in os.listdir(r"C:\Users\WX\Desktop\brick\picture"):  #使用os.listdir()获取该文件夹下每一张图片的名字
    im=imageio.imread(filename)


    i=i+1
    np.save(filename,im)
    if(i==100):   #190为文件夹中的图片数量
        break

