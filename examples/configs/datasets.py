dataset_defaults = {

    'iwildcam': {
        'loss_function': 'cross_entropy',
        'val_metric': 'F1-macro_all',
        'model_kwargs': {'pretrained': True},
        'transform': 'image_base',
        'target_resolution': (448, 448),
        'val_metric_decreasing': False,
        'algo_log_metric': 'accuracy',
        'model': 'resnet50',
        'lr': 3e-5,
        'weight_decay': 0.0,
        'batch_size': 16,
        'n_epochs': 12,
        'optimizer': 'Adam',
        'split_scheme': 'official',
        'scheduler': None,
        'groupby_fields': ['location',],
        'n_groups_per_batch': 2,
        'irm_lambda': 1.,
        'coral_penalty_weight': 10.,
        'no_group_logging': True,
        'process_outputs_function': 'multiclass_logits_to_pred',
        'process_pseudolabels_function': 'pseudolabel_multiclass_logits',
    },

}

