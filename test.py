import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms


def predict_img(image):
    image_tensor = test_trainsforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor).to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def get_random_images(num):
    data = datasets.ImageFolder(val_dir, transform=test_trainsforms)
    classes = data.classes

    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]

    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)

    images, labels = dataiter.next()
    return images,labels,classes


if __name__ == '__main__':
    val_dir = './data/val/'
    test_trainsforms = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor(), ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('./save_model/model.pth')
    model.eval()
    to_pil = transforms.ToPILImage()
    images, labels, classes= get_random_images(3)
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(images)):
        image = to_pil(images[i])
        index = predict_img(image)
        sub = fig.add_subplot(1, len(images), i+1)
        res = int(labels[i]) == index
        str(sub.set_title(str(classes[index])+":"+str(res)))
        plt.axis('off')
        plt.imshow(image)
    plt.show()