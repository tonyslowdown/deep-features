- Data
  - Download dog images
	- ImageNet http://vision.stanford.edu/aditya86/ImageNetDogs/
  - Download and curate korean food images
	- Go through each photo and remove irrelevant ones

- TODO
  - Take tensorflow course on udacity, watch some of Richard Socher's videos online (stanford course) - 1 week, 7 lectures, some twice
  - Prepare deep features data - modify inception v3 classify_image.py to extract deep features, and run it on all image datasets
  - Install tensorflow w gpu acceleration - https://medium.com/@fabmilo/how-to-compile-tensorflow-with-cuda-support-on-osx-fd27108e27e1#.gi1hidszm
	- apply fix described here before compiling: https://github.com/tensorflow/tensorflow/issues/582
	- Please specify the Cuda SDK version you want to use. [Default is 7.0]: 7.5
	- Please specify the Cudnn version you want to use. [Default is 6.5]: 4
	- Please specify a list of comma-separated Cuda compute capabilities you want to build with. [Default is: "3.5,5.2"]: 3.0
	- speedup on benchmarked w inception v3 retrain:
	  - 2000 training steps: 2m18.055s -> 2m30.295s no speedups at all
	  - 1000 training steps: 1m11.341s -> 1m19.523s
  - Transfer learning
	- using deep features
	  - other sklearn classifiers
	  - Upload same train-test set to metamind and test how it does using api
	  - tensorflow retraining - https://www.tensorflow.org/versions/master/how_tos/image_retraining/index.html#training-on-your-own-categories
  - Retrain entire v3 network? - not for udacity capstone (need server, laptop will fry)
  - Caffe pretrained networks - not for udacity capstone (no time)
  - Create android app that recommends restaurants - yelp api?

- Tech stack
  - android - submit for udacity capstone
  - ios app?
  - cassandra/lucene?
  - angular2.0?
  - web server?
	- django
	- golang

- Possible applications:
  - Korean food recommender system using deep features
  - Lost dogs, children, stolen cars
  - Use deep features as embedding, like wordvec (find source pdf to cite)
	- filtering images in feeds, i.e. instagram
	- reverse - let users know that their image is not that special
	- pca to reduce features (find meaning of each attribute from papers) - didn't work
	  - second row: puppy, clothes, person in the picture, which are more interesting than first row (image_processing.ipynb)
	  - PCA with marginal features (http://imagine.enpc.fr/~aubrym/projects/features_analysis/texts/understanding_deep_features_with_CG.pdf)
		- various colors
		- square sizes
		- different cropped shapes of same image

- Report: write arxive-type paper in markdown
  - formulate/motivate the problem
  - optimize computing - inception modules same goal (look at paper http://arxiv.org/pdf/1512.00567v3.pdf)
  - set benchmark accuracy with sklearn algorithms + metamind
  - analyze test accuracy, confusion matrix, etc using tensorflow results similar to metamind results https://www.metamind.io/classifiers/41180/stats
  - similarity ranking - http://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf
	- how to measure? - use confusion matrix to see which ones are mistaken the most
	- check that deep feature embedding distances is monotonic with respect to "likeness" - proof?
		- show cluster of tSNE
		- show avg distance from each other vs other labels
  - BUG: pandoc has issues with having a figure and a table on the same page - it'll push the text on the page down below the margin:
    http://tex.stackexchange.com/questions/276699/the-longtable-environment-pushes-content-below-it-into-the-bottom-margin-of-a-pa

# Generate Report

``` bash
pandoc report.md -o report.pdf --filter pandoc-citeproc --latex-engine=xelatex
```
