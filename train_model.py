import torchvision.transforms as transforms
from utils.data_builder import create_dataloaders
from VIT import VisionTransformer
from datetime import datetime
import os
import torch
import torch.nn as nn
from utils.engine import train
from utils.util import plot_loss_curves

#HYPERPARAMETERS
img_size = 224
batch_size = 32
num_epochs = 2
num_heads = 6
num_layers = 12
img_transforms = transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()])
train_dir, test_dir = "data/train", "data/test"
model_name = f"VisionTransformer_{num_layers}_{num_heads}"
experiment_name = "foodvision_mini"
train_save_dir = "checkpoints"
learning_rate = 1e-4
weight_decay = 0.05



if __name__ == "__main__":

    train_loader, test_loader, class_names = create_dataloaders(train_dir=train_dir,
                                                                test_dir=test_dir,
                                                                transform=img_transforms,
                                                                batch_size=batch_size)
    
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    experiment_dir = os.path.join(train_save_dir, model_name, experiment_name)
    working_dir = os.path.join(experiment_dir, timestamp)
    os.makedirs(working_dir, exist_ok=True)

    num_classes = len(class_names)

    model = VisionTransformer(
        img_size=img_size,
        in_channels=3,
        patch_size=16,
        num_layers=num_layers,
        embedding_dim=768,
        mlp_size=3072,
        num_heads=num_heads,
        attn_dropout=0.0,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
        num_classes=num_classes
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
    
    results = train(model=model,
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    optimizer=optimizer,
                    scheduler=schedular,
                    loss_fn=criterion,
                    epochs=num_epochs,
                    save_dir=working_dir,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    writer=None) #type:ignore
    
    plot_loss_curves(results, working_dir)

    