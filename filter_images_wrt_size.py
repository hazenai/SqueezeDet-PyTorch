import os
import sys
from PIL import Image
image_path = '/workspace/SqueezeDet-PyTorch_simple_bypass/data/kitti/training/image_2'
image_path=os.path.abspath(image_path)
imageIds = os.listdir(image_path)
print(len(imageIds))
imgIdNames=[]
for imgId in imageIds:
    img = Image.open(os.path.join(image_path, imgId))
    if img.size[0] >= 200 and img.size[1] >= 200:
        imgIdNames.append(imgId)
with open("./filteredImages_size>=200_and.txt", 'w') as fp:
    for idimg in imgIdNames:
        fp.write("%s\n" % idimg[:-4])
