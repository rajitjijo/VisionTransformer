import torch
from torchvision import models
import os, glob
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def pred_and_display(images, transforms, model):

    class_names = ["pizza", "steak", "sushu"]

    for image in images:
        img = Image.open(image).convert("RGB")
        img_transformed = transforms(img)
        img_transformed = img_transformed.unsqueeze(0)

        with torch.inference_mode():
            preds = model(img_transformed)
            pred_cls = class_names[preds.argmax(dim=1).item()]

        # true label = folder name
        true_cls = os.path.basename(os.path.dirname(image))

        # show image with matplotlib
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Pred: {pred_cls} | True: {true_cls}")
        plt.show()


if __name__ == "__main__":

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=3) #type:ignore
    transforms = models.ViT_B_16_Weights.DEFAULT.transforms()
    test_images = glob.glob("data/test/**/*.jpg", recursive=True)
    pred_and_display(test_images, transforms, model)