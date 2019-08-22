# Conditional VAE 

## In short
This is a Tensorflow 1.14 implementation of the CVAE model from [Sohn, Kihyuk, Xinchen Yan, and Honglak Lee. "Learning Structured Output Representation using Deep Conditional Generative Models."](https://pdfs.semanticscholar.org/3f25/e17eb717e5894e0404ea634451332f85d287.pdf). 

A short explanation of the CVAE model and formulation of the losses can be found in [this notebook](https://github.com/Nocty-chan/CVAE/blob/master/CVAE%20implementation.ipynb)

The model is trained on binarized MNIST dataset with three types of conditioning:
  * Class conditioning - (See Class Conditional VAE notebook)
  * Upper-left quadrant - (See Quadrant Conditional VAE notebook)
  * Half image quadrant
 
 ## Results

 ![Class conditioning](https://user-images.githubusercontent.com/13089230/63478259-fd1b8880-c43d-11e9-9f6c-3c6bf0ffbc17.png)
 ![UL quadrant](https://user-images.githubusercontent.com/13089230/63478260-fd1b8880-c43d-11e9-8255-776f6b196d23.png)
 ![Half](https://user-images.githubusercontent.com/13089230/63478261-fd1b8880-c43d-11e9-876f-5d34f57d6e9f.png)

Left: Logit results for CVAE conditioned on class. Each column corresponds to one class with different latent vectors.

Middle: Logit results for CVAE conditioned on a quadrant. First row shows the image each sample is conditioned on. Notice the variation in the digit for the same conditioning (first row has 0s and 4s).

Right: Logit results for CVAE conditioned on a half image. First row shows the image each sample is conditioned on. 

## Usage

Please refer to the notebooks for how to run the different models. 
Model training and evaluation can also be directly launched from the `main.py` file.
