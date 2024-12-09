import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

# 图像识别列表（寓言、童话、科学、历史、民俗）
BLIP2_xl = [70.0,58.1, 65.0, 58.8, 71.2,                 64.3]
BLIP2_xxl = [71.2,55.8,  70.0, 63.7, 63.6,                  64.8]
InstructBLIP_xl = [65.0,70.9,  77.5, 68.8, 71.2,            70.7]
InstructBLIP_xxl = [69.8, 70.2,  74.5, 69.2, 70.6,           70.8]

# 设置柱状图的位置和宽度
x = np.arange(len(InstructBLIP_xxl))  # 使用numpy生成一个数组
width = 0.15

# 绘制柱状图
fig, ax = plt.subplots(figsize=(13, 6))  # 设置一个更大的画布
rects1 = ax.bar(x - 1.6*width, BLIP2_xl, width, label='BLIP-2 XL', color='lightgreen')
rects2 = ax.bar(x - 0.5*width, BLIP2_xxl, width, label='BLIP-2 XXL', color='green')
# rects3 = ax.bar(x + 0.6*width, InstructBLIP_xl, width, label='InstructBLIP XL', color='lightgreen')
# rects4 = ax.bar(x + 1.7*width, InstructBLIP_xxl, width, label='InstructBLIP XXL', color='green')

rects3 = ax.bar(x + 0.6*width, InstructBLIP_xl, width, label='InstructBLIP XL', color='skyblue')
rects4 = ax.bar(x + 1.7*width, InstructBLIP_xxl, width, label='InstructBLIP XXL', color='deepskyblue')

ax.set_ylim(50, 80)

# 添加一些文本标签
# ax.set_ylabel('Binary Accuracy (%)')
# ax.set_title('Identification')
ax.set_xticks([])  # 隐藏原有的x轴刻度


# 在纵坐标的每个刻度上添加标线
for ytick in ax.get_yticks():
    ax.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, zorder=0)

# 添加图例
ax.legend(handlelength=1, handleheight=1, ncol=2, columnspacing=0.5, handletextpad=0.5, labelspacing=0.1)

# 显示图形
plt.tight_layout()  # 调整布局以适应图像
plt.show()
plt.savefig('./identification.png')


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import numpy as np


# # 设置全局字体大小
# plt.rcParams.update({'font.size': 18})

# # 图像解释列表（寓言、童话、科学、历史、民俗）
# BLIP2_xl = [12.5, 16.1, 7.5, 10.9,7.5,           10.7]
# BLIP2_xxl = [ 7.5, 12.9, 15.0, 26.1, 22.5,             17.3]
# InstructBLIP_xl = [ 55.0,29.0,57.5, 28.3, 55.0,           45.2]
# InstructBLIP_xxl = [56.1, 31.2,60.1, 28.5, 54.0,        46.0]

# # 设置柱状图的位置和宽度
# x = np.arange(len(InstructBLIP_xxl))  # 使用numpy生成一个数组
# width = 0.15

# # 绘制柱状图
# fig, ax = plt.subplots(figsize=(13, 6))  # 设置一个更大的画布
# rects1 = ax.bar(x - 1.6*width, BLIP2_xl, width, label='BLIP-2 XL', color='lightgreen')
# rects2 = ax.bar(x - 0.5*width, BLIP2_xxl, width, label='BLIP-2 XXL', color='green')
# # rects3 = ax.bar(x + 0.6*width, InstructBLIP_xl, width, label='InstructBLIP XL', color='lightgreen')
# # rects4 = ax.bar(x + 1.7*width, InstructBLIP_xxl, width, label='InstructBLIP XXL', color='green')

# rects3 = ax.bar(x + 0.6*width, InstructBLIP_xl, width, label='InstructBLIP XL', color='skyblue')
# rects4 = ax.bar(x + 1.7*width, InstructBLIP_xxl, width, label='InstructBLIP XXL', color='DeepSkyBlue')

# ax.set_ylim(5, 65)

# # 添加一些文本标签
# # ax.set_ylabel('GPT4 Rating (%)')
# # ax.set_title('Explanation')
# ax.set_xticks([])  # 隐藏原有的x轴刻度

# # 在纵坐标的每个刻度上添加标线
# for ytick in ax.get_yticks():
#     ax.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5, zorder=0)

# # 添加图例
# ax.legend(handlelength=1, handleheight=1, ncol=2, columnspacing=0.5, handletextpad=0.5, labelspacing=0.1)

# # 显示图形
# plt.tight_layout()  # 调整布局以适应图像
# plt.show()
# plt.savefig('./explanation.png')
