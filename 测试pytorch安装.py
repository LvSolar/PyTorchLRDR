import torch
import torchvision


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')
    print("HELLO pytorch {}".format(torch.__version__))
    print("torchvision.version:", torchvision.__version__)
    print("torch.cuda.is_available? ", torch.cuda.is_available())

    x = torch.rand(3, 6)
    print(x)

# import torch
# from torch.autograd import Variable
# import torch.nn.functional as Func
# import matplotlib.pyplot as plt
#
# # 生成数据序列
# tensor = torch.linspace(-5, 5, 200)  # 返回一维张量
# tensor = Variable(tensor)  # Tensor 转为 Variable
# xNp = tensor.numpy()  # Tensor 转为 numpy
#
# # 定义激活函数
# y_relu = torch.relu(tensor).data.numpy()
# y_sigmoid = torch.sigmoid(tensor).data.numpy()
# y_tanh = torch.tanh(tensor).data.numpy()
# y_softplus = Func.softplus(tensor).data.numpy()
#
# # 绘图
# plt.figure(figsize=(9, 6))
# plt.suptitle("Response curve of activation function")
# plt.subplot(221)
# plt.plot(xNp, y_relu, c='red', label='RelU')
# plt.legend(loc='best')
# plt.subplot(222)
# plt.plot(xNp, y_softplus, c='red', label='hardTanh')
# plt.legend(loc='best')
# plt.subplot(223)
# plt.plot(xNp, y_sigmoid, c='red', label='sigmoid')
# plt.legend(loc='best')
# plt.subplot(224)
# plt.plot(xNp, y_tanh, c='red', label='tanh')
# plt.legend(loc='best')
# plt.show()
