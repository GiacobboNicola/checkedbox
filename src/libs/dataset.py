import glob
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


class BoxesDataset(Dataset):
    def __init__(self, root_dir, in_gray, transform=None, format=".jpg", dims=(45, 45)):
        self.root_dir = root_dir
        self.transform = transform
        self.in_gray = in_gray
        self.img_dim = dims
        self.data = []
        self.class_map = {}

        classes_path = glob.glob(self.root_dir + "*")

        for id, class_path in enumerate(sorted(classes_path)):
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*" + format):
                self.data.append([img_path, class_name])
            self.class_map[class_name] = id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = T.PILToTensor()(pil_img)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        return img_tensor.float(), class_id
