Extract BDD100K detection images and box annotations under `data/bdd100k/` (gitignored),
for example from the [Kaggle mirror](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k/).
Both official train and val images and detection JSONs are required; images or
segmentation data alone are insufficient. Start with `notebooks/01_data_preparation.ipynb`.
See the root README for canonical layout, toolkit compatibility, class mapping,
coordinate conventions and split/exposure rules. Never commit data or weights.
