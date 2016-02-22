- Download images
  - Download dog images
	- ImageNet http://vision.stanford.edu/aditya86/ImageNetDogs/
	- Instagram https://gist.github.com/rubinovitz/6348163
  - check another couple of the 1000 classification categories in imagenet
  - one category for non-classification category on imagenet (show that this works in imagenet)

- Modify inception v3 classify_image.py to extract deep features, and run it on all image datasets

- Technology
  - cassandra/lucene
  - angular2.0?
  - set up caffe, torch
  - use dato, caffe pretrained networks

- search for other “filtering” methods to compare against

- applications:
  - use transfer learning for filtering categories not included in imagenet
  - reverse - let users know that their image is not that special

- pca to reduce features (find meaning of each attribute from papers)

- Report: write arxive-type paper in markdown
  - use deep features as embedding, like wordvec (find source pdf to cite)
  - predict action: 2 approaches
	- PCA with marginal features (http://imagine.enpc.fr/~aubrym/projects/features_analysis/texts/understanding_deep_features_with_CG.pdf)
	- training another layer to predict action
  - formulate/motivate the problem
  - optimize computing - inception modules same goal (look at paper http://arxiv.org/pdf/1512.00567v3.pdf)
  - similarity ranking - http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf
	- how to measure?

second row:
puppy
clothes
with somebody (human)

marginal analysis
	- various colors
	- square sizes
	- different cropped shapes of same image
