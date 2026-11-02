Download BDD100K images **and detection box annotations**, for example from the
[Kaggle mirror](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k/),
and extract them under `data/bdd100k/` (gitignored). Nested Kaggle directories are
supported. Both official train and val image/label splits are required.

Images or segmentation data alone are insufficient. Check for
`bdd100k_labels_images_train.json` and `bdd100k_labels_images_val.json`, or equivalent
Scalabel det_train/val labels containing `box2d`. Some mirrors have only a small
annotated subset; `vd survey` reports usable coverage and missing files.

See the root README for the category mapping, validation rules, split controls,
installation and reproducible commands. Never commit dataset files or weights.
