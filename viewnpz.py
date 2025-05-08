import numpy as np
from PIL import Image
import os

data = np.load('results/VIRTUAL_lsun_bedroom256.npz')

# 查看其中的数组名称（键名）
print(data.files)  # 通常为 ['arr_0'] 或其他

images = data['arr_0']
print(images.shape)

# os.makedirs('results/sample_256x256_classifier/output_images', exist_ok=True)
#
# # 保存前几张图片，条件生成
# for i in range(10):
#     img = Image.fromarray(images[i].astype(np.uint8))
#     img.save(f"results/sample_256x256_classifier/output_images/sample_{i}_{data['arr_1'][i]}.png")
#
# # 无条件生成
# for i in range(10):
#     img = Image.fromarray(images[i].astype(np.uint8))
#     img.save(f'results/sample_256x256_classifier/output_images/sample_{i}.png')
