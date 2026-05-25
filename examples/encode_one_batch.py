"""Lower-level example: load one BioPM and encode a single batch of windows.

Useful if you have already preprocessed your data into PyTorch tensors and
just want to call the encoder directly (no DataLoader needed).
"""

import torch
import biopm

# 1.  Load the 50MR pretrained model
model = biopm.load_pretrained(masking_rate=0.5, device="cpu")

# 2.  Build a *fake* batch with the shapes BioPM expects.
#     In a real workflow these tensors come from `biopm.load_preprocessed_h5`.
B, L = 4, 192          # batch of 4 windows, up to 192 movement elements each
patches = torch.randn(B, L, 32)                  # normalised ME patches
patches[:, 50:] = float("nan")                   # mark the last rows as padding
pos_info = torch.linspace(0, 1, L).expand(B, L)  # fractional position
additional_emb = torch.zeros(B, L, 2)            # channel 0 = axis id, 1 = duration
additional_emb[:, :, 0] = torch.randint(0, 3, (B, L)).float()
additional_emb[:, 50:] = float("nan")
gravity = torch.randn(B, 300, 3)                  # 3-axis gravity window

# 3.  Encode -> default per-axis fused feature (B, 1023)
features = biopm.encode_window(
    model, patches, pos_info, additional_emb, gravity,
    per_axis=True,
)
print("per-axis feature shape:", features.shape)

# 4.  Same thing with the legacy total mean+std pooling (B, 1028)
features_total = biopm.encode_window(
    model, patches, pos_info, additional_emb, gravity,
    per_axis=False,
)
print("total   feature shape:", features_total.shape)
