# Embedding Visualization

This is an example of Embedding Visualization using T-SNE

## How to works
Run HTTP server:

> `python -m http.server`

- Then, go to: http://localhost:8000/ to view your visualization

## Try with your dataset

1. Prepare *.tsv files

*.tsv have 2 files:
- meta.tsv: contains labels of all images
- vecs.tsv: conntains vector embedding of all images

In this examples, I use face images, and vector embedding is created by [InsightFace pretraind models](https://github.com/deepinsight/insightface). 

2. Prepare sprites image
Sprites image is created by concatation all your input images.
View `create_sprites.py` files and try to RUN it:
> `python create_sprites.py`
