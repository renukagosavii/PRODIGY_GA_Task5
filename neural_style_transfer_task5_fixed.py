import os
import torch
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError

# ---------- Paths ----------
content_path = r"C:\Users\gosav\OneDrive\Desktop\PRODIGY_GA_05\circle.jpg"
style_path   = r"C:\Users\gosav\OneDrive\Desktop\PRODIGY_GA_05\starry_night.jpg"
   # or .png
output_path  = r"C:\Users\gosav\OneDrive\Desktop\PRODIGY_GA_05\output.jpg"

# ---------- Function to safely load image ----------
def load_image_safe(path, max_size=400):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        exit()
    try:
        img = Image.open(path).convert('RGB')
    except UnidentifiedImageError:
        print(f"❌ Cannot open image (corrupted or wrong format): {path}")
        exit()
    # Resize
    size = max_size if max(img.size) > max_size else max(img.size)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return transform(img).unsqueeze(0)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load images ----------
content = load_image_safe(content_path).to(device)
style   = load_image_safe(style_path).to(device)

# ---------- Helper functions ----------
def gram(tensor):
    c, h, w = tensor.size()[1:]
    tensor = tensor.view(c, h*w)
    return torch.mm(tensor, tensor.t())

def get_features(x, model, layers={'0':'conv1_1','5':'conv2_1','10':'conv3_1','19':'conv4_1','21':'conv4_2','28':'conv5_1'}):
    feats = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            feats[layers[name]] = x
    return feats

# ---------- Load pre-trained VGG19 ----------
vgg = models.vgg19(pretrained=True).features.to(device).eval()
content_feat = get_features(content, vgg)
style_feat   = get_features(style, vgg)
style_grams  = {l: gram(style_feat[l]) for l in style_feat}

# ---------- Initialize target ----------
target = content.clone().requires_grad_(True).to(device)
optimizer = torch.optim.Adam([target], lr=0.003)

# ---------- Weights ----------
style_weights = {'conv1_1':1.0,'conv2_1':0.75,'conv3_1':0.2,'conv4_1':0.2,'conv5_1':0.2}
content_weight, style_weight = 1e4, 1e2

# ---------- Training ----------
print("Starting style transfer...")
for i in range(1, 301):
    feats = get_features(target, vgg)
    c_loss = torch.mean((feats['conv4_2'] - content_feat['conv4_2'])**2)
    s_loss = sum(style_weights[l]*torch.mean((gram(feats[l])-style_grams[l])**2) for l in style_weights)
    loss = content_weight*c_loss + style_weight*s_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}/300, Loss: {loss.item():.2f}")

# ---------- Save output ----------
final = target.detach().cpu().squeeze()
final_img = transforms.ToPILImage()(final.clamp(0,1))
final_img.save(output_path)
print(f"✅ Style transfer done! Saved as: {output_path}")




