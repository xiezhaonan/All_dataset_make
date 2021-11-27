"""This module contains simple helper functions """
import os
from sklearn.metrics import confusion_matrix


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


# def generate_label_text(path):
#
#     image_lists = os.listdir(path)  # 返回path下所有文件的列表
#     if 'labels.txt' in image_lists:
#         os.remove(path + '\labels.txt')
#         image_lists.remove("labels.txt")
#     print(image_lists)
#     count = 0
#     context_list = []
#     with open(path + '\labels.txt', 'a+') as f:
#         while True:
#             if image_lists != []:
#                 del (image_lists[0])
#             else:
#                 break
#             label_list = []
#             for img_name in image_lists:
#                 # print(img_name)
#                 if img_name.split('_')[0] == 'F':
#                     label = 1
#                 elif img_name.split('_')[0] == 'N':
#                     label = 0
#                 label_list.append(label)
#                 # context = '{path}/{image_name},{label},'.format(path=path, image_name=img_name, label=label)
#                 context = '{path}/{image_name},'.format(path=path,
#                                                         image_name=img_name)
#                 context_list.append(context)
#                 count += 1
#                 if count % 20 == 0:
#                     label_set = set(label_list)
#                     temp1 = ""
#                     for temp in context_list:
#                         temp = str(temp)
#                         temp1 += temp
#                     if len(label_set) == 1 and label == 1:
#                         print("debug_label", label)
#                         # context_list.append("\n")
#                         context_str = temp1 + '1' + '\n'
#                     elif len(label_set) == 1 and label == 0:
#                         context_str = temp1 + "0" + "\n"
#                     else:
#                         context_list = []
#                         break
#                     # filename = os.path.split(file)[0]
#                     # filetype = os.path.split(file)[1]
#                     # print(filename, filetype)
#                     # if filetype == '.txt':
#                     #     continue
#                     # name = '/teeth' + '/' + file + ' ' + str(int(label)) + '\n'
#                     print(context_str)
#                     f.write(context_str)
#                     context_list = []
#                     break
#     print("finished!")

# 根据不同的模式使用不同的生成标签的文件
import numpy as np
import random
import os
def generate_label_text_for_nine_classes(path):
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    class8 = []
    seed = 2021
    random.seed(seed)

    image_lists = os.listdir(path)  # 返回path下所有文件的列表
    if 'labels.txt' in image_lists:
        os.remove(path + '\labels.txt')
        image_lists.remove("labels.txt")
    print(image_lists)
    with open(os.path.join(path, 'labels.txt'), "a+") as f:
        for name in image_lists:
            print(name)
            index = int(name.split(',')[0])
            print(index)
            if index == 0:
                class0.append((name, 0))

            elif index == 1:
                class1.append((name, 1))

            elif index == 2:
                class2.append((name, 2))
            elif index == 3:
                class3.append((name, 3))
            elif index == 4:
                class4.append((name, 4))
            elif index == 5:
                class5.append((name, 5))
            elif index == 6:
                class6.append((name, 6))
            elif index == 7:
                class7.append((name, 7))
            elif index == 8:
                class8.append((name, 8))
            f.writelines(f"{os.path.join(path, name)},{index}\n")
    #
    # random.shuffle(class1)
    print("finished!")


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """

    if not os.path.exists(path):
        os.makedirs(path)


def confusion_visual(confusion_matrix,lens):
    import numpy as np
    import matplotlib.pyplot as plt
    if lens < 10:
        classes = [str(0)+str(i)  for i in range(lens)]
    else:
        assert lens < 19, 'value error'
        classes = [str(1) + str(i) for i in range(lens)]
    print(classes)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i, j] for j in range(9)] for i in range(9)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]), horizontalalignment="center")  # 显示对应的数字
    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig("confusion matrix visual.png")
    plt.show()



if __name__ == "__main__":
    generate_label_text_for_nine_classes(r'D:\Pytorchexample\自定义数据集练习\标签在图片名字上在一起\data')