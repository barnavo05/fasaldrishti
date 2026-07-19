"""
FasalDrishti — AI Produce Grading & Shelf Life Estimation
Built on first-author IEEE research (AMLDS 2026): CNN + DCFNet + Grad-CAM.
"""

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ----------------------------- Page config -----------------------------
st.set_page_config(page_title="FasalDrishti", page_icon="🍎", layout="centered")

# ----------------------------- Model definition ------------------------
# Identical to the architecture in the IEEE paper notebook (FruitSenseNet).

class DCFLayer(nn.Module):
    def __init__(self, in_channels, spatial=14, sigma=2.0, lam=0.01):
        super().__init__()
        self.spatial, self.lam = spatial, lam
        coords = torch.arange(spatial).float() - (spatial - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        g = g / (g.sum() + 1e-8)
        self.register_buffer("gaussian", g.unsqueeze(0).unsqueeze(0))
        self.out_dim = in_channels

    def forward(self, x):
        b, c, h, w = x.shape
        if (h, w) != (self.spatial, self.spatial):
            x = F.adaptive_avg_pool2d(x, (self.spatial, self.spatial))
            h = w = self.spatial
        g = self.gaussian.to(x.device).expand(b, c, h, w)
        f_hat = torch.fft.rfft2(x)
        g_hat = torch.fft.rfft2(g)
        denom = (f_hat.conj() * f_hat).real + self.lam
        h_filter = (f_hat.conj() * g_hat) / denom
        response = torch.fft.irfft2(f_hat * h_filter, s=(h, w))
        return response.mean(dim=[2, 3])


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class FruitSenseNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.block5 = ConvBlock(256, 512)
        self.dcfnet = DCFLayer(in_channels=128, spatial=14)
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat = 512 + self.dcfnet.out_dim
        self.fc1 = nn.Linear(feat, 256)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        dcf = self.dcfnet(x)
        x = self.block4(x)
        x = self.block5(x)
        pooled = self.gap(x).flatten(1)
        h = torch.cat([pooled, dcf], dim=1)
        h = self.drop1(F.relu(self.fc1(h)))
        h = self.drop2(F.relu(self.fc2(h)))
        return self.fc3(h)


# ----------------------------- Constants -------------------------------
CLASS_NAMES = ["Fresh", "Ripe", "Rotten"]  # must match training class_to_idx order
IMG_SIZE = 224
WEIGHTS_PATH = "model/fasaldrishti_weights.pt"

# Shelf life mapping (class + confidence heuristic, documented in README)
SHELF_LIFE = {
    "Fresh": ("5–7 days", "Good for storage and long-distance shipping."),
    "Ripe": ("2–4 days", "Sell soon — prioritize local/nearby markets."),
    "Rotten": ("0–1 day", "Not fit for sale. Remove to protect nearby produce."),
}
GRADE_COLOR = {"Fresh": "#1B5E20", "Ripe": "#E65100", "Rotten": "#B71C1C"}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----------------------------- Model loading ---------------------------
@st.cache_resource
def load_model():
    model = FruitSenseNet(num_classes=len(CLASS_NAMES))
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------- Grad-CAM --------------------------------
def gradcam(model, input_tensor, class_idx):
    """Grad-CAM on the last conv block (block5)."""
    activations, gradients = [], []

    def fwd_hook(_m, _i, out):
        activations.append(out)

    def bwd_hook(_m, _gi, grad_out):
        gradients.append(grad_out[0])

    h1 = model.block5.register_forward_hook(fwd_hook)
    h2 = model.block5.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(input_tensor)
    logits[0, class_idx].backward()

    h1.remove()
    h2.remove()

    acts = activations[0].detach()[0]          # (C, H, W)
    grads = gradients[0].detach()[0]           # (C, H, W)
    weights = grads.mean(dim=(1, 2))           # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy()


def overlay_heatmap(pil_img, cam, alpha=0.45):
    """Blend a jet-style heatmap over the original image using pure numpy/PIL."""
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(pil_img.size, Image.BILINEAR)
    cam_arr = np.asarray(cam_img, dtype=np.float32) / 255.0

    # Simple jet colormap: blue -> green -> red
    r = np.clip(1.5 - np.abs(4 * cam_arr - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_arr - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_arr - 1), 0, 1)
    heat = np.stack([r, g, b], axis=-1)

    base = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))


# ----------------------------- UI --------------------------------------
st.title("🍎 FasalDrishti")
st.caption(
    "AI produce grading with explainable heatmaps and shelf-life estimates. "
    "Built on first-author IEEE research (AMLDS 2026)."
)

tab_upload, tab_camera = st.tabs(["📤 Upload photo", "📷 Use camera"])
img_file = None
with tab_upload:
    img_file = st.file_uploader("Upload a fruit photo", type=["jpg", "jpeg", "png"])
with tab_camera:
    cam_file = st.camera_input("Photograph the produce")
    if cam_file is not None:
        img_file = cam_file

if img_file is not None:
    image = Image.open(img_file).convert("RGB")

    with st.spinner("Analyzing..."):
        model = load_model()
        x = TRANSFORM(image).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        idx = int(probs.argmax())
        label = CLASS_NAMES[idx]
        conf = float(probs[idx])
        cam = gradcam(model, TRANSFORM(image).unsqueeze(0), idx)
        heat_img = overlay_heatmap(image, cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(heat_img, caption="Why? (Grad-CAM)", use_container_width=True)

    color = GRADE_COLOR[label]
    st.markdown(
        f"<h2 style='color:{color};'>Grade: {label} "
        f"<span style='font-size:0.6em;color:#666;'>({conf:.0%} confidence)</span></h2>",
        unsafe_allow_html=True,
    )

    life, advice = SHELF_LIFE[label]
    st.markdown(f"**🕒 Estimated shelf life:** {life}")
    st.markdown(f"**💡 Recommendation:** {advice}")

    st.markdown("---")
    with st.expander("All class probabilities"):
        for name, p in zip(CLASS_NAMES, probs.tolist()):
            st.progress(p, text=f"{name}: {p:.1%}")

    st.caption(
        "The heatmap highlights the image regions that most influenced the grade "
        "(bruising, discoloration, texture change). Shelf-life shown is a "
        "class-and-confidence based estimate; see README for methodology."
    )
else:
    st.info("Upload a photo or use your camera to grade produce. Sample images are in the GitHub repo.")
