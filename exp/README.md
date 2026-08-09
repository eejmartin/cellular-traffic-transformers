# exp — experiment class

`Exp_Main` (`exp_main.py`) is the single experiment class for all four
models: train / validate / test / predict run through the same loop so
results stay directly comparable. Model dispatch happens in `MODEL_CLASSES`
and `_model_forward()` (Peri-midFormer additionally receives time-feature
marks and its `label_len+pred_len` output is sliced to `pred_len`; the other
three take only the input series). Exactly one learning-rate schedule is
active per run, selected by `args.lradj` (`'TST'` = OneCycleLR per batch,
`'cos'` = warmup + cosine for DeformableTST's paper recipe, otherwise
epoch-level decay). `seed_everything()` here makes every training
reproducible — it is re-applied per station. `Exp_Basic` (`exp_basic.py`)
holds device selection and the shared skeleton.
