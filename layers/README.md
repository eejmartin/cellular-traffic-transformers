# layers — shared building blocks

Neural building blocks reused by the models in `models/`:

| File | Contents |
|---|---|
| `PatchTST_backbone.py`, `PatchTST_layers.py` | patch embedding + channel-independent Transformer encoder used by PatchTST |
| `SelfAttention_Family.py`, `Global_Attn.py`, `Transformer_Module.py` | attention variants (full, deformable helpers) and encoder blocks |
| `PerimidFormer_EncDec.py` | periodic-pyramid encoder/decoder used by Peri-midFormer |
| `Embed.py` | value / positional / time-feature embeddings |
| `RevIN.py` | reversible instance normalization (per-window normalize → denormalize) |
| `head.py` | flatten prediction heads |
