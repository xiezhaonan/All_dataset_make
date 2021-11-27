import torch
import os, glob
import random, csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CSV_Mydataset(Dataset):
    def __init__(self, root, resize, mode):
        super(CSV_Mydataset, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}  # "sq...":0
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                # print('1')
                continue

            self.name2label[name] = len(self.name2label.keys())

        print(self.name2label)

        # self.images, self.labels = self.load_csv('images.csv')
        self.images, self.labels = self.load_csv('images.csv')



        if mode == 'train':  ####  60%  traning
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'val':  ####  60%-80%  test
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6*len(self.images)):int(0.8*len(self.images))]
        else: #  ####  80%-100%  valuation
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels = self.labels[int(0.8*len(self.labels)):]




        ####    image, label
    def load_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                ####    'pokemon\\mewywo\\00001.png'
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images), images)
            print('1')

            ####  1167, 'pokemon\\mewywo\\00001.png'

            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:  #########  'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    ####  'pokemon\\mewywo\\00001.png', 0
                    writer.writerow([img, label])
                print('writrn into csv flie', filename)


        #####  read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
        ####  'pokemon\\mewywo\\00001.png', 0
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels



    def __len__(self):
        return  len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x_hot = (x-mean)/std
        # x= x_hot*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)        ###########   Normalize 逆操作  图片还原
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean

        return x


    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),#   string path => image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
            transforms.RandomRotation(15),   #####   旋转15度
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)


        return  img, label


def main():


    import  visdom
    import  time
    import  torchvision

    viz = visdom.Visdom()

    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x,y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))   #############   简单的一行代码进行数据的加载和编码
    #
    #     time.sleep(10)

    db = CSV_Mydataset('data', 224, 'train')

    x, y = next(iter(db))
    print('sample', x.shape, y.shape, y)
    viz.image(x, win='sample_x', opts=dict(title='sample_x'))
    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))  #####  还原图像
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)


    for x, y in loader:
        viz.images(db.denormalize(x),  win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        time.sleep(2)


if __name__ == '__main__':
    main()
