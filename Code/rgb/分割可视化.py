import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np
from PIL import Image
from skimage.segmentation import slic
import math
from IPython import display

#rgb_path = 'X://Project\myproject//data//rgb//Indian_pines_rgb.jpg'
#rgb_path = 'X://Project//myproject\data//rgb//Salinas_rgb.jpg'
rgb_path = 'X://Project//myproject\data//rgb//PaviaU_rgb.jpg'
img = Image.open(rgb_path)
img_array = np.array(img)
print(img_array.shape)
h = img_array.shape[0]
w = img_array.shape[1]
print(h, w)
block = 5
# The number of superpixel
n_superpixel = int(math.ceil((h * w) / block))
seg = slic(img_array, n_superpixel, 10)
# 使用mark_boundaries函数来标记分割的边界
marked_img = mark_boundaries(img_array, seg)
# 创建一个新的图像
plt.figure(figsize=(10,10))
# 显示图像
plt.imshow(marked_img)
plt.axis('off')  # 关闭坐标轴
display.clear_output(wait=True)
# 保存图片为矢量图格式，没有坐标轴
plt.savefig(f"PU—block={block}.svg", format='svg', bbox_inches='tight', pad_inches=0)
plt.close()
