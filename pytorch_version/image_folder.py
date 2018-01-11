import torchvision.transforms as transforms
import torch.utils.data as data
import os

from PIL import Image

# The qualified suffix name
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset(dir):

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                # create the label
                if fname[0:3] == 'cat':
                    target = 0
                elif fname[0:3] == 'dog':
                    target = 1
                # create a turple containing the path of image and the target
                item = (path, target)
                images.append(item)
    return images


def default_loader(path):
    # load image from path using PIL
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """
    ImageFolder for loading image
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):

        imgs = make_dataset(root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):

        return len(self.imgs)


def testmyImageFloder():
    """ Test """
    transform_train = transforms.Compose([
        transforms.Resize((300, 300)),
    ])

    dataloader = ImageFolder('./catdog_test', transform=transform_train)
    print(dataloader)
    for index, (img, label) in enumerate(dataloader):
        img.show()
        print('label', label)

        if index >= 3:
            break


# if __name__ == "__main__":
#     testmyImageFloder()