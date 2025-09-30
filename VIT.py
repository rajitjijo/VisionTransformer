import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dim:int=768, image_size:int=224):

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        assert self.image_size % self.patch_size == 0; f"Input Image cannot be broken into {self.patch_size} patches"
        self.number_of_patches = int((self.image_size**2) / (self.patch_size**2))
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embedding_dim, stride=self.patch_size, kernel_size=self.patch_size)
        self.flat = nn.Flatten(start_dim=-2, end_dim=-1)
        self.class_token = nn.Parameter(torch.randn(1,1,self.embedding_dim))
        self.position_token = nn.Parameter(torch.randn(1,self.number_of_patches+1,self.embedding_dim))

    def forward(self, x):
        
        """
        x(input) is of shape -> (batch_size, channels, img_height, img_width)
        output is of shape -> (batch_size, num_of_patches + 1, embedding_dim)
        """

        batch_size = x.shape[0]
        cls_token = self.class_token.expand(batch_size,-1,-1)
        x = self.conv(x)
        x = self.flat(x)
        x = x.permute(0,2,1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_token

        return x

class AttentionEncoder(nn.Module):

    def __init__(self, num_heads:int=1, embedding_dim:int=768, mlp_dropout:float=0.1, attn_dropout:float = 0.0, dense_dim:int=3072):

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.attn_dropout = attn_dropout
        self.dense_dim = dense_dim
        self.mlp_dropout = mlp_dropout
        self.layers = nn.ModuleDict({
            "attention": nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=self.attn_dropout, batch_first=True),
            "layer_norm1": nn.LayerNorm(normalized_shape=self.embedding_dim),
            "layer_norm2": nn.LayerNorm(normalized_shape=self.embedding_dim),
            "linear1": nn.Linear(in_features=self.embedding_dim, out_features=self.dense_dim),
            "linear2": nn.Linear(in_features=self.dense_dim, out_features=self.embedding_dim)
        })
        self.dropout1 = nn.Dropout(p=self.mlp_dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        x(input) is of shape -> (batch_size, num_of_patches + 1, embedding_dim)
        output is of shape -> (batch_size, num_of_patches + 1, embedding_dim)
        """
        #Attention Block
        x_norm = self.layers["layer_norm1"](x)
        attended_x, _ = self.layers["attention"](x_norm, x_norm, x_norm, need_weights=False)
        x = x + attended_x #skip 2
        #FeedForward Block
        attended_x_norm = self.layers["layer_norm2"](x)
        attended_x_norm = self.layers["linear1"](attended_x_norm)
        attended_x_norm = self.gelu(attended_x_norm)
        attended_x_norm = self.dropout1(attended_x_norm)
        attended_x_norm = self.layers["linear2"](attended_x_norm)

        return x + attended_x_norm

class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_layers:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=1,
                 attn_dropout:float=0.0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000
                 ):
        
        super().__init__()
        self.image_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.embedding_dropout = embedding_dropout
        self.num_classes = num_classes

        self.embedding = PatchEmbedding(in_channels=self.in_channels, patch_size=self.patch_size, embedding_dim=self.embedding_dim, image_size=self.image_size)

        self.attention_encoder = nn.ModuleList([AttentionEncoder(
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            mlp_dropout=self.mlp_dropout,
            attn_dropout=self.attn_dropout,
            dense_dim=self.mlp_size
        ) for _ in range(self.num_layers)])

        self.embedding_dropout_layer = nn.Dropout(self.embedding_dropout)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=self.num_classes)
        )

    def forward(self, x):

        x = self.embedding(x)
        x = self.embedding_dropout_layer(x)
        for layer in self.attention_encoder:
            x = layer(x)
        x = self.classifier(x[:,0])
        return x



if __name__ == "__main__":

    pass
    # random_img = torch.rand(1,3,224, 224).to("cuda")

    # embed = PatchEmbedding()

    # print(f"Shape of embedded images {embed(random_img).shape}")
    
    # for name, param in embed.named_parameters():
    #     print(name, param.shape, param.requires_grad)

    # total = sum(p.numel() for p in embed.parameters() if p.requires_grad)
    # print(f"Total trainable params: {total}")

    # encoder = AttentionEncoder()

    # print(encoder(random_img).shape)

    # for name, param in encoder.named_parameters():
    #     print(name, param.shape, param.requires_grad)

    # desc = summary(
    #     model=encoder,
    #     input_size=(200,3,224,224),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"]
    # )

    # print(str(desc))

    # vit = VisionTransformer(num_classes=3).to("cuda")
    # logits = vit(random_img)              # tensor on GPU
    # probs = torch.softmax(logits, dim=1)  # softmax in PyTorch
    # print(probs.detach().cpu().numpy())
    # print(np.argmax(probs.detach().cpu().numpy()))   # convert only for printing
    
    