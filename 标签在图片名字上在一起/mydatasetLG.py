'''
Author: your name
Date: 2021-01-25 14:29:10
LastEditTime: 2021-01-25 14:29:10
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \modify_module\mydataset.py
'''
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from util import generate_label_text_for_nine_classes
# txt_path='data'
class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()  # 去除txt文档中每行结尾的指定字符（默认是空格）
            words = line.split()    # 将每行数据以空格分开
            # print(words[0],words[1])
            imgs.append((words[0], int(words[1]))) # 将图片地址 和对应的label以元组的形式放在一个列表中

            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]        # 将存储在列表中的图像地址 和 label 分开
        # print(fn)
        # img = Image.open(fn).convert('RGB')
        img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)





# class CnnLstmDataset(Dataset):
#
#     def __init__(self, txt_path, transform=None):
#         self.txt_path = txt_path
#         self.transform = transform
#         self.image_list = []
#         with open(self.txt_path, 'r') as f:
#             for line in f:
#                 images = [img_path for img_path in line.split(',')]
#                 self.image_list.append((images[:-1], images[-1]))
#
#     def __getitem__(self, index):
#         #print(self.image_list[index])
#         X, labels = self._concat_images(self.image_list[index])
#         # print('debug', X.shape, labels)
#         #print('debug11 labels,int(labels)', labels, int(labels))
#         # labels = torch.LongTensor(int(labels))
#         labels = torch.tensor(int(labels), dtype=torch.float32)
#         #print('debug11 labels',labels)
#         return X, labels
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def _concat_images(self, frames):
#         X = []
#
#         for i in range(len(frames[0])):
#             # print('debug0:', frames[0][i])
#             # print('debug1:', frames[1])
#             image = Image.open(frames[0][i]).convert('RGB')
#             # print('debug2:',image)
#             if self.transform is not None:
#                 image = self.transform(image)
#                 # print('Debug3:',image)
#             X.append(image)
#         # print('debug4:', X)
#         X = torch.stack(X, dim=0)
#         #print(f'X shape:{X.size()}')
#         return X, frames[1]

def main():
    import  visdom
    import  time
    viz = visdom.Visdom()
    db = MyDataset('data', None, None)
    x, y = next(iter(db))
    print('sample', x.shape, y.shape, y)
    viz.image(x, win='sample_x', opts=dict(title='sample_x'))
    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))  #####  还原图像
    loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=8)


    for x, y in loader:
        # viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(2)


if __name__ == '__main__':
    main()
        # 看近处的物体为0 看远处的物体为1
        # folder_name = ['train', 'val', 'test','test_ZS']
        # folder_name = ['exp']
        # img_path = r'E:\Intelligent_Wheelchair\dataset'
        # for i in range(len(folder_name)):
        #     use_path = os.path.join(img_path, folder_name[i])
        #     generate_label_text_for_nine_classes(use_path)

        # todo 根据不同的分类任务，使用不同的制作txt文档的函数
        # folder_name = ['train', 'val', 'test']
        # img_path = r'.\data'
        # generate_label_text_for_nine_classes(img_path)

