from dataset import CelebAMask
from torch.utils.data import DataLoader
import torchvision
from matplotlib import pyplot as plt

root_dir = "/home/yinzi/data3/CelebAMask-HQ"
# DataLoader
celba = CelebAMask(root_dir=root_dir, mode='test')

dataloader = DataLoader(celba, batch_size=10,
                        shuffle=False, num_workers=4)

for batch in dataloader:
    image = batch['image']
    label = batch['labels']
    image_grid = torchvision.utils.make_grid(image)
    label_arg =label.argmax(dim=1, keepdim=False)
    label_grid = torchvision.utils.make_grid(label_arg.unsqueeze(dim=1))
    plt.imshow(image_grid.permute(1, 2, 0))
    plt.pause(0.01)
    plt.imshow(label_grid[0])
    plt.pause(0.01)

