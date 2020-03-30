# Embedding Visualization

This is an tiny example for Embedding Visualization using T-SNE

## How to works
Run HTTP server:

> `python -m http.server`

- Then, go to: http://localhost:8000/ to view your visualization

## Try with your dataset

### 1. Prepare *.tsv files

*.tsv have 2 files:
- meta.tsv: contains label of all images
- vecs.tsv: contains vector embedding of all images

In this examples, I use face images, and vector embedding is created by [InsightFace pretrained models](https://github.com/deepinsight/insightface). 

### 2. Prepare sprites image
Sprites image is created by concatenating all your input images.

Look at `create_sprites.py` file & try to RUN it:

> `python create_sprites.py`
