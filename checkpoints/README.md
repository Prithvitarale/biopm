# Pretrained BioPM checkpoints

Three pretrained variants of the BioPM acc-stream encoder
(`biopm.model.TimeSeriesTransformer`) ship in this directory:

| File             | Masking rate | Notes                  |
| ---------------- | ------------ | ---------------------- |
| `biopm_25mr.pt`  | 25%          |                        |
| `biopm_50mr.pt`  | 50%          | default; used in paper |
| `biopm_75mr.pt`  | 75%          |                        |

Each file is a plain PyTorch state dict containing the weights of the
acc-stream encoder only (no classifier head, no gravity CNN).

Load via the convenience API:

```python
import biopm
model = biopm.load_pretrained(masking_rate=0.5)   # default
```

or directly:

```python
import torch
from biopm import BioPM
m = BioPM()
m.load_checkpoint("checkpoints/biopm_50mr.pt")
m.eval()
```
