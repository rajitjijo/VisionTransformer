# VisionTransformer
Implementing the Vision Transformer research paper

**Input to a vision transformer**

```
x_input = [class_token, image_patch_1, image_patch_2,.....,image_patch_n] + 
[class_token_pos, image_patch_1_pos,........,image_patch_n_pos]
```

__Example Patched Image__

This is the 16x16 patched image that passes onto the multiheaded attention layer
![](asset/patched_pizza.png)

