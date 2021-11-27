import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from  torchvision import transforms
from    PIL import Image
from MyDataset import CSV_Mydataset


# path = 'D:\Pytorchexample\自定义数据集联系\标签在图片名字上在一起\data'
path = '.\data'



class MyDataset(Dataset):                        #########   继承dataset
    def __init__(self, path, transform=None):  ###########   继承一些属性
        self.path = path                        #########     文件路径
        self.transform = transform              ##########    对图形进行处理
        self.images = os.listdir(self.path)     ##########    把路径下的所有文件放在一个列表中
        # self.resize = resize
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(path))):
            if not os.path.isdir(os.path.join(path, name)):
                continue

            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)
        self.images, self.labels = self.load_csv('images.csv')

    def __len__(self):                           #########    返回数据集的大小

        return len(self.images)


    def __getitem__(self, index):              #############  根据索引index返回图像及标签

        image_index = self.images[index]       #############   根据索引获取图像文件名称
        image_path = os.path.join(self.path, image_index)  ####    获取图像的路径或目录
        img = Image.open(image_path).convert('RGB')       ####    读取图像

        tf = transforms.Compose([
            # lambda x: Image.open(x).convert('RGB'),  # string path => image data
            # transforms.Resize(224),
            # transforms.RandomRotation(15),  #####   旋转15度
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        #######################    根据目录名称获得图像的标签 ####  #####  绝对引用
        label = image_path.split('\\')[-1].split('.')[0]

        ######################    把字符转换为数字    dog->1  else->0
        # label = 1 if 'dog' in label else 0

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.path, filename)):
            images = []
            for name in self.name2label.keys():
                ####    'pokemon\\mewywo\\00001.png'
                images += glob.glob(os.path.join(self.path, name, '*.png'))
                images += glob.glob(os.path.join(self.path, name, '*.jpg'))
                images += glob.glob(os.path.join(self.path, name, '*.jpeg'))
            print(len(images), images)

            ####  1167, 'pokemon\\mewywo\\00001.png'

            random.shuffle(images)
            with open(os.path.join(self.path, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  #########  'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    ####  'pokemon\\mewywo\\00001.png', 0
                    writer.writerow([img, label])
                print('writrn into csv flie', filename)


        #####  read from csv file
        images, labels = [], []
        with open(os.path.join(self.path, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
        ####  'pokemon\\mewywo\\00001.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels







def main():

        images = os.listdir(path)
        print(images)
        len(images)
        dataset = MyDataset(path, transform= None)
        img, label = next(iter(dataset))
        print('1')


        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(loader)

        # for batch_datas, batch_labels in loader:
        #     print(batch_datas.size(),batch_labels())
        #     break



if __name__ == '__main__':
    main()






















