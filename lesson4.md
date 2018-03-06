Dropout
=======

A dropout layer goes through an X% chance of being deleted (set to 0). The
output only changes by a little bit if we do that, since its only one piece
of data that makes up the entire activation on the next layer. For each
mini-batch we throw away another random subset of the layer. It forces the
network to find a representation that continues to work even as random
parts of the activations get thrown away.

It has just about solved the problem of generalization for us. If you tried
to train a model with lots of parameters and you were overfitting, to
a large degree you were kind of stuck.

Now, when we look at the validation set, you obviously turn off dropout - you
want to be using the best model you can. So our validation accuracy and loss
tends to be better overall.

Also note that when you throw away half of the activations, you need to double
the value of the activations that are remaining. Basically multiply
by 1 / p.

Not really any rules of thumb as to where to apply dropout. Some people
say only apply dropout on the last layer.

Overfitting
===========

Note that more parameters means more chance to overfit (because you can
approximate a function to the data better!). So you'll need to use more
dropout in those cases.

To determine whether or not overfitting is happening, you can compare
your training loss to your validation loss. If the training loss is going
down by the validation loss is going up, you are overfitting!

Adding additional fully connected layers
========================================

You can also pass `extra_fc` to add additional fully connected layers
asides from the mandatory one that goes from your output activations
to your classes.

Structured and Time Series Data
===============================

There's columns that we think of categorical which has levels and then there
are continuous columns. Categorical columns need to be one-hot encoded.

The thing is that sometimes it works better to treat continuous things as
categorical - some things look continuous but each element is going to behave
qualitatively differently.

You can also split continuous variables into ranges and then make it categorical
from there. Obviously have sensible upper and lower bounds if you want to do
this - you don't want a matrix that is 2014 rows long.

Question: What happens when you start treating things like years as categories
and you end up testing against a year that you've never seen before? Short
answer is that it gets treated as an unknown category. It'll still predict
something. If there's any unknowns in the raining set it'll figure out
a way to predict unknown.

Scaling
=======

Neural nets really like to have the input data have a mean of zero and a
standard deviation of one. So usually you'll want to normalize the data
that you get from a strucutred dataset.

Creating Validation Sets
========================

Obviously you need to look at the problem in order to work out what a good
validation set actually is. For instance, in the Groceries competition, you're
asked to predict the *next two weeks* of sales, so you don't just want random
outcomes for your validation set.

What is your metric
===================

Sometimes your metric won't be in fastai itself. For instance, root mean squared
error is not in there.

```
sum(y / y') / n
```

Embeddings
==========

For our continuous variables, all we're going to do is grab them all and
get a bunch of floating point numbers. The neural net takes that vector
and put it through a matrix multiplication having as many rows as
the tensor and as many columns as outputs.

What about categorical variables? Say you have `DayOfWeek` and one-hot encode it
with length 7. How do we feed that in so that we still end up with a rank-1
tensor of floats?

We create a new little matrix of 7 rows and as many columns as we choose
(say 4). We do a lookup into the matrix and we grab the relevant row
(eg, "Sunday"s particular 4 floating point numbers). So we convert Sunday
into a rank-1 tensor of 4 floating point numbers. The whole matrix starts
out random. Note that the "lookup" is the same thing as multiplying by
a one-hot encoded vector by the matrix.

We put it through the neural net and at the very end we find out the loss and
do gradient descent to improve the matrix (so we update the four numbers in
the matrix for "Sunday"). The matrix is another bunch of weights in our Neural
Net. This is called an "Embedding Matrix".

The row gets appended to the end of our continous variables (as opposed to
the one-hot encoded version of the category). What is the advantage of that?
Just using one-hot encoded variables could work, but the problem is that
it gets linear behaviour. These embedding vectors get richer semantic concepts -
for instance, the network will learn that Sunday and Saturday are related
by being a weekend (in relation to sales). So instead of working with
categories like "Saturday" and "Sunday", the Network is actually working
with "Weekend" or "Gas Sales".

What is a good heuristic for the dimensionality of the embedding matrix? Make
a little list of every categorical variable and its cardinality. For instance,
1000+ different stores, 7 days of the week, 4 years etc. The rule of thumb is
to take the cardinality of the variable, divide it by 2, don't make it
bigger than 50.

An embedding is suitable in any categorical variable. The only thing it
can't work well for is something that is too high cardinality. You'd have
to bucketize it. Everything that is not too high cardinality you can just
make a categorical variable and make it an embedding.

What about Seasonality (dates and times)? There's a fastai function called
`add_datepart`, which replaces the date string with the components. That
way we end up with a lot more features. Conceptually we can create some
pretty interesting timeseries models with that. The only thing is that
your cyclic thing needs to exist as a column and not just as a categorical
variable.

```
min(50, cardinality(variable) / 2)
```

You would typically do an Embedding Matrix for each categorical variable.

Creating a learner
==================

You can create a `ModelData` directly out of a pandas data frame - you can
use the `ColumnarModelData.from_data_frame` function. Arguments:
 - `path`: Where to store on disk
 - `val_idxs`: Validation indices
 - `df`: Independent variables
 - `yl`: log of Dependent variable
 - `cat_flds`: What fields are treated as categories?
 - `bs`: Batch size

Now, to create the learner: `model.get_learner`:
 - `emb_szs`: These are the embeddings' sizes.
 - `continuous_n`: Number of continuous variables.
 - `dropout_embedding`: Dropout that gets applied to the embedding matrix
 - `activations`: List of activation numbers.
 - `linear_layer_dropouts`: List of dropouts for linear layers.
 - `y_range`: Range of dependent variable.

Then just go ahead and start training. Note that in this case we parse
in a metric which is the exponential root mean squared error.

Can we use data augmentation in this case?
==========================================

No idea. Has to be domain specific. Don't know if you can do it
with structured data.

What is dropout doing? Exactly the same as before - trying to prevent
overfitting.

What's the downside of doing it this way? Nobody in Academia is working
on structured data problems because they're boring. Until now, there
hasn't been any way to do it conveniently.

Natural Language Processing
===========================

This is the area that is definitely up and coming area. A lot of the stuff
you'll see in NLP is derived from Computer Vision reseach.

Language Modelling: Build a model, where given a few words of a sentence,
can you predict the next word of the sentence. Example: Download 18 months
worth of papers from arxiv.org. Pass the model some "priming text", eg
"<CAT> csni <SUMM> algorithms that".

The model starts our not knowing English with an embedding matrix for every
word in English that was random.

Of course, we don't care about this at all. What we're actually going to
do is take IMDB movie reviews and work out whether they are positive or
negative. We'd really like to use a pre-trained network that *knows how to
read english*. To know how to read english, you should be able to predict
the next word of the sentence. Instead of predicting the next word of the
sentence, we can instead use the model for classification.

Why can't you just tune the embeddings from the data directly? You can't
really learn the entire structure of English from a 1 or a 0 doesn't really
work - instead you teach it to reach English first.

Language models tend to work at a word-level, of course that doesn't deal
with typos or new words.

Torchtext
=========

Torchtext is pytorch's NLP library.

Creating a language model
=========================

Before we can do anything with text, we have to turn it into tokens. You'll
notice that the tokenizer will do a good job at recognizing pieces of
an English sentence (wasn't -> was n't). spaCy is pretty good at that.

Now, first we create a torchtext fieldwhich describes how to preprocess a
piece of text -> `data.Field(lower=True, tokenize=spacy_tok)`. What we can
now do is go ahead and create the usual fastai model data object. Need
to provide:
 - `training_path`
 - `validation_path`
 - `test_path`
 - `preprocessor`: The `data.Field` above
 - `bs`: Batch size
 - `bptt`:
 - `min_freq`: If there is any word that occurrs less than X times, ignore it.
 - `bptt`: "Backprop through time" - how long a sentence we have on the GPU.

Now, once the `LanguageModelData` is created, we have a `TEXT.vocab` object,
which maps integers to strings.

Allows us to match a word to an integer or an integer to a word.

Is it common to do any stemming or lemmatizing? Generally, no. Tokenization
is really all that is necessary.

When you deal with natural language is context important? Yes, very! The
bag-of-words approach is no longer useful - we're starting to recognize how
to use deep learnign to recognize context properly.

Now, `backprop through time`: Even though we have lots of movie reviews, they
all get concatenated together. We split it up into batches first, for instance
if we want a batch size of 64, then we break the 60 million words into
64 sections and we move each section "underneath" the previous one. So we end
up with a matrix that is 60x10^6/64 x 64. We then grab a little chunk of this
at a time which is approximately equal to `bptt`. That's a batch. Each bit
is a sequence. Note that each epoch, you're getting slightly different windows,
of slightly different lengths. This is sort of the equivalent of shuffling
images.

What if instead of just arbitrarily choosing 64? Why not choose sentences and
then pad with zero (so that you have one full sentence per line?). Now
although it is true that the columns don't exactly stop on the end of
sentence, it doesn't matter because there are *so* many sentences.

Training
========

The 35,000 unique words is used to create an embedding - you're just
creating a very very high cardinality cardinality variable.

Note that when we do create the embedding, the row length is going to
be quite high - 200 in this case. This is because each word has
quite a lot of nuance to it.

We then also have to say how many activations you want in your layers
which in this case is 500 and the number of layers (3).

We can get a leaner from our model by calling `md.get_model`. We use
something called AWD LSTMs which different amounts of dropouts in different
places. If you're underfitting, decrease all of the dropouts and if you're
overfitting, increase all of the dropouts.

Another trick is clip the learning rate when using SGDR - it prevents
bouncing around too much. Go ahead and call `.fit` with exactly
the same parameters as usual.

Note that the loss will probably be pretty bad, but you can check the
model by getting it to predict text and see how well it performs.

What about word2vec and GloVe
=============================

Basically, people have pre-trained these embedding matrices and you
can download them. There's no reason we couldn't download them, but
Jeremy has found that building whole pre-trained model from using
word2vec didn't really help much more than just pre-training your
own language model from the actual data.


Classification on Sentiment
===========================

Now if we're going to use a pre-trained model we still need to use
the classification. We need to split up all the text into all
of the bits.

Fastai has a library called `TextData.from_splits` to pass in
chunks of text and a classification.

Once we've creaeted the mdoel, we can load into it the pre-trained
language model as the bottom layers through `load_encoder`. Once
you've got a pre-trained language model it actually trains super
fast.

Collaborative Filtering
=======================

We're using the movielens dataset. User #1 watched movie #31 and gave
it a rating of 3. Our goal is that for some user-movie combination
we haven't seen before, we have to predict whether they'll like it.

To make it more interesting we'll download a list of movie. We're
creating a cross-product of users and movies.