import torchvision.transforms as transforms
from utils.data_builder import create_dataloaders

#HYPERPARAMETERS
img_size = 224
batch_size = 32
train_dir, test_dir = "data/train", "data/test"
img_transforms = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])





if __name__ == "__main__":

    train_loader, test_loader, class_names = create_dataloaders(train_dir=train_dir,
                                                                test_dir=test_dir,
                                                                transform=img_transforms,
                                                                batch_size=batch_size)
    

    print(len(train_loader))