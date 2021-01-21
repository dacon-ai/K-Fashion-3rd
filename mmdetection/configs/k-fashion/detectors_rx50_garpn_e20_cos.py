_base_ = [
    './detectors_rx50_garpn.py',
]

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)

total_epochs = 20
