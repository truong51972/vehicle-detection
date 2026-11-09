Extract BDD100K detection images and box annotations under `data/bdd100k/` (gitignored),
for example from the [Kaggle mirror](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k/).
Both official train and val images and detection JSONs are required; images or
segmentation data alone are insufficient. Start with
[00_data_init.ipynb](../notebooks/00_data_init.ipynb). `configs/data.yaml` stores the
project-relative `data_root`. Never commit data or weights.
