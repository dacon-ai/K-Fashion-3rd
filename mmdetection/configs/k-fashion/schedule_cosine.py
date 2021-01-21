# optimizer
# optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)

total_epochs = 20
