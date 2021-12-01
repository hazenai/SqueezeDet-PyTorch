import skimage.io as skio
from time import time
iters = 100
stime = time()
for i in range(iters):
    im = skio.imread('../data/LPR/training/image_2/aiactive_000005.jpg')
etime = time()
s = 'SKIO: It took {}ms to load {} images. {}FPS'.format((etime-stime)*1000, iters, iters/(etime-stime))
print(s)
print('-'*25)
# ---------------------------------------------------------------------
import cv2
stime = time()
for i in range(iters):
    im = cv2.imread('../data/LPR/training/image_2/aiactive_000005.jpg')
etime = time()
s = 'CV2: It took {}ms to load {} images. {}FPS'.format((etime-stime)*1000, iters, iters/(etime-stime))
print(s)
print('-'*25)
# ---------------------------------------------------------------------
from PIL import Image
coord = x, y = 1, 1
stime = time()
for i in range(iters):
    im = Image.open('../data/LPR/training/image_2/aiactive_000005.jpg')
    im.getpixel(coord)
etime = time()
s = 'PIL: It took {}ms to load {} images. {}FPS'.format((etime-stime)*1000, iters, iters/(etime-stime))
print(s)
print('-'*25)
