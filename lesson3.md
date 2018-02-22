Thinking in Keras
=================

In Keras we have a 'data generator', where oyu sepcify the data augmentation and
what kind of batch normalization is required. You have to know about what is expected
by the model.

You the take the data generator and create an actual generator which flows from
a directory. You have to create a data generator for the validation set that does
not shuffle and does not augment data.

You need to construct a base model and then put the layers you want to add on top
of them (usually average pooling and a dense layer on top).

Then you have to freeze and unfreeze layers manually, then compile the model by
specifying an optimizer, a loss functon and a metrics evaluator.

Dimensionality of Images
========================

Usually images are three dimensional WxHx3, though sometimes PyTorch needs
four dimensional matrices because it operates on mini-batches. To fix that, use
`image[None]` to index `None` into the matrix and add an additional axis.

(eg, you now have a matrix of WxHx3x1)

CNN Image Intro
===============

A convolution is something where we have a little matrix (3x3) and multiply
each element in that matrix by a pixel, sum them together, then replace the pixel
with the value.

For instance, say you have a filter that looks like this:

-1 0 1
-1 0 1
-1 0 1

So everywhere where the white edge is matching the edge of the letter, you're
getting a positive. Everywhere it doesn't, you get a negative. Then you throw
away the negatives.

Then do another filter:

1  1  1
0  0  0
-1 -1 -1

Same thing, and throw away the negatives again.

Then in layer 4, you replace every 2x2 part of a grid with its maximum.

So you have two filtered max-pooled images.

Next, you're going to be applying each new filter to both filtered
max pooled images. So the output of filter N is the sum of
the application of filter N to max pooled images 1 to M. Eg, you
have an Mx3x3 kernel

Repeat until the max pooled images are of a certain size.

Then compare it to a learned template.

What if you have three channels of input?

You would have 3x3x4 filters to start with, then 3x3x4xM

Why is it that edge-detecting convolutions actually detect edges?
=================================================================

Consider a filter like the bottom-edge detector:

```
 1  1  1
 0  0  0 
-1 -1 -1
```

If we run this over something that is a bottom edge, for instance:

```
1.0 1.0 1.0     1.0 1.0 1.0
0.8 0.8 0.8  -> 0.0 0.0 0.0 -> 3.0
0.1 0.0 0.0    -0.1 0.0 0.0
```

What ends up happening is that the bottom bits which were zero anyway
stay zero, the top bits get highlighted and the middle bits go to zero.

If we run it over a top edge, the opposite happens:

```
0.0 0.0 0.0    0.0  0.0  0.0
0.4 0.5 0.9 -> 0.0  0.0  0.0 -> -3.0
1.0 1.0 1.0   -1.0 -1.0 -1.0
```

To rectify it, just do:

`MAX(0, activation)`

Fully Connected Layer
=====================

We have our max pooling which combines into a fully connected layer. This
is the classic linear algebra matrix product.

Now, for MNIST, instead of wanting one set of fully connected weights, we
would have ten of those.


Activations
===========

Now, we want to take all of our outputs and we want to scale them so that
they are all between 0 and 1 and add up to 1.

Softmax
-------

Always spits out numbers between 0 and 1 and always spits out numbers
that add to 1. In theory, this isn't strictly necessary - it could learn
a set of kernels that give probabilities that line up as closely as we want. But
if you can construct your architecture that allows you to express what you want as intuitively as possible, you get better models.

Take the sum of all the exponentials of the activations.

Then divide each exponential by the sum of the exponentials.

The exponential pushes out the slightly larger numbers further the
softmax minimizes the small differences.


Sigmoid
-------

If a thing can have multiple classes, then softmax isn't going to work.
Softmax wants to pick **a thing** (remembering that all the probabilities
need to add up to one). So if your thing has two labels, the highest probability
for each is going to be 0.5. 0.33 for three labels and so on.

Instead we use something called the sigmoid function which is just:

```
e^x / (1 + e^x)
```

Analytically we can see that this also ends up between zero and one, though
not all the values have to sum up to one. It sort of follows an s-shape, where
negative values are pushed towards zero and positive values are pushed
towards 1 (note that at x = 0, we have 1 / 2, so for all x > 0, 0.5 < S(x) <
1.0). It also has that nice property where separates big values out from
smaller ones more significantly, considering that e^x grows exponentially.

However, if you have lots of large x values, then remember that
lim (e^x / (1 + e^x)) -> 1, so those things will become indistinguishable.


Differential Learning Rates and Layer Freezing
==============================================

Recall that in a CNN you have convolutional layers and dense layers. Usually
you'll be starting with pre-trained convolutional layers trained on ImageNet
which are designed to recognise common features in photos. The dense layers
are either Xavier-initialized or randomly initialized and are utter garbage.

It stands to reason then that you'll want to train the dense layers a little
bit more, then train the convolutional layers (but not as much, since they
already have useful information in them).

So it is pretty common practice to use the pretrained convolutional models
and just train the dense layers to start with. Once we have fitted the
dense layers, we then unfreeze the convolutional layers.

However, we don't want to train the convolutional layers at the same rate
as the dense layers, because otherwise you're just going to bounce
around the loss space. So this is where **differential learning rates** come in.

The fastai library allows us to set different learning rates for each layer
or sets of layers depending on how divisible they are. This is done by
passing an array of learning rates.

What learning rates you end up passing really depends on how wrong we think
the earlier layers are. For image recognition tasks like dog breeds, the
convolutional layers already know how to recognize photographic features
very well, so there's very little need to adjust them. The pooled layers
might need a little bit of adjustment to focus more on dogs, so we
have a slightly higher learning rate there. Then the dense layers
will need a lot of training, so we use our base learning rate.

For tasks such as recognising medical or cosmic images we'll need much
higher learning rates for the convolutional and pooling layers, though
not necessarily as much as the dense layers.
