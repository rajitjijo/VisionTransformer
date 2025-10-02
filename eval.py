import torch
from torchvision import models
import os, glob
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from time import time
import gradio as gr

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

model = models.vit_b_16(weights=None)
model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=3) #type:ignore
transforms = models.ViT_B_16_Weights.DEFAULT.transforms()
model.load_state_dict(torch.load("checkpoints/VisionTransformer_B16_Pretrained/foodvision_mini/02_10_2025_12_25_10/train50/train50.pth"))

def predict(img):

    class_names = ["pizza", "steak", "sushi"]

    start_time = time()
    # img = Image.open(img).convert("RGB")
    img = transforms(img)
    img = img.unsqueeze(0)

    model.eval()

    with torch.inference_mode():

        pred_probs = torch.softmax(model(img), dim=1)

    pred_labels_and_probs = {class_names[i]:round(pred_probs[0][i].item(),2) for i in range(len(class_names))}

    pred_time = round(time() - start_time, 4)

    return pred_labels_and_probs, pred_time

title = "FoodVisionMini🍕🥩🍣"
desc = "Uses a vision transformer architecture to classify images of food into either steak, pizza or sushi"

demo = gr.Interface(fn=predict, inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3,label="Predictions"), gr.Number(label="Predition Time (s)")],
                    title=title, description=desc)


if __name__ == "__main__":

    # test_images = glob.glob("data/test/**/*.jpg", recursive=True)
    # pred_and_display(test_images, transforms, model)

    demo.launch(share=True, debug=False)