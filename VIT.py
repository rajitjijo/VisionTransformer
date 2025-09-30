import torch
import torch.nn as nn

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
        self.class_token = nn.Parameter(torch.zeros(1,1,self.embedding_dim))
        self.position_token = nn.Parameter(torch.zeros(1,self.number_of_patches+1,self.embedding_dim))

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

    def __init__(self, num_heads:int=1, embedding_dim:int=768, in_channels:int=3, patch_size:int=16, image_size:int=224, dropout:float = 0.0, dense_dim:int=3072):

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.dropout = dropout
        self.dense_dim = dense_dim
        self.layers = nn.ModuleDict({
            "embedding": PatchEmbedding(self.in_channels, self.patch_size, self.embedding_dim, self.image_size),
            "attention": nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads, dropout=0, batch_first=True),
            "layer_norm1": nn.LayerNorm(normalized_shape=self.embedding_dim),
            "layer_norm2": nn.LayerNorm(normalized_shape=self.embedding_dim),
            "linear1": nn.Linear(in_features=self.embedding_dim, out_features=self.dense_dim),
            "linear2": nn.Linear(in_features=self.dense_dim, out_features=self.embedding_dim)
        })
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        x(input) is of shape -> (batch_size, channels, img_height, img_width)
        output is of shape -> (batch_size, num_of_patches + 1, embedding_dim)
        """
        #Firstly converting our batch of images into patched embeddings
        x = self.layers["embedding"](x) #skip 1
        #MHA Pass
        x_norm = self.layers["layer_norm1"](x)
        attended_x, _ = self.layers["attention"](x_norm, x_norm, x_norm, need_weights=False)
        del x_norm
        attended_x = x + attended_x #skip 2
        #Linear Pass
        attended_x_norm = self.layers["layer_norm2"](attended_x)
        attended_x_norm = self.layers["linear1"](attended_x_norm)
        attended_x_norm = self.gelu(attended_x_norm)
        attended_x_norm = self.dropout1(attended_x_norm)
        attended_x_norm = self.layers["linear2"](attended_x_norm)

        return attended_x + attended_x_norm


if __name__ == "__main__":

    random_img = torch.rand(200,3,224,224)

    # embed = PatchEmbedding()

    # print(f"Shape of embedded images {embed(random_img).shape}")
    
    # # for name, param in embed.named_parameters():
    # #     print(name, param.shape, param.requires_grad)

    # total = sum(p.numel() for p in embed.parameters() if p.requires_grad)
    # print(f"Total trainable params: {total}")

    encoder = AttentionEncoder()

    # print(encoder(random_img).shape)

    for name, param in encoder.named_parameters():
        print(name, param.shape, param.requires_grad)