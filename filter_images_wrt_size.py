import os
import sys
from PIL import Image
# image_path = '/workspace/SqueezeDet-PyTorch_simple_bypass/data/kitti/training/image_2'
image_path = '/workspace/SqueezeDet-PyTorch_simple_bypass/data/kitti/training/synth_180k/image_2'
image_path=os.path.abspath(image_path)
imageIds = os.listdir(image_path)
print(len(imageIds))
imgIdNames=[]
for imgId in imageIds:
    img = Image.open(os.path.join(image_path, imgId))
    if img.size[0] >= 300 or img.size[1] >= 300:
        imgIdNames.append(imgId)
with open("./filtered_Images_size>=300_or_synth_180k.txt", 'w') as fp:
    for idimg in imgIdNames:
        fp.write("%s\n" % idimg[:-4])
