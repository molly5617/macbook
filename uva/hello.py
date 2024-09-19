import cv2
import matplotlib.pyplot as plt

# 读取图片
image = cv2.imread('lena.bmp')

# 检查图片是否成功加载
if image is None:
    print("无法加载图片。请确保 lena.bmp 文件在正确的路径下。")
else:
    # 将 BGR 格式转换为 RGB 格式（因为 OpenCV 默认使用 BGR）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 显示图片
    plt.imshow(image_rgb)
    plt.title('Lena')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()

    # 保存为另一个文件
    cv2.imwrite('lena_output.bmp', image)