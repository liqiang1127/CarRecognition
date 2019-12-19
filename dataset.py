from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):  # 因为漏了这行代码，花了一个多小时解决问题
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloaders(data_dir, ratio, batchsize=32):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:]:
            val_inputs.append(str(x))
            val_labels.append(i)
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, data_transforms['train']), batch_size=batchsize,
                                  drop_last=True, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, data_transforms['valid']), batch_size=batchsize,
                                drop_last=True, shuffle=True)
    loader = {}
    loader['train'] = train_dataloader
    loader['valid'] = val_dataloader
    return loader


if __name__ == '__main__':
    data_dir = '/home/lab205/WorkSpace/liqiang/compitition/image'
    """ 每一类图片有1300张，其中780张用于训练，260张用于测试，260张用于测试"""
    loader = fetch_dataloaders(data_dir, [0.8, 0.2], batchsize=32)
    for image, label in loader['valid']:
        print(label.shape)