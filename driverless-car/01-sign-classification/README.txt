# Writeup

The goal of this project is to recognize traffic signs. The general
approach is to sue a series of convolutions, alternated with max
pooling, followed by a couple fully connected layers.

## Dataset Exploration

You can see my basic dataset exploration in the notebook. I first
looked at some basic summary statistics about the number of examples
and classes. I also called up examples of each of the traffic signs,
displaying these so I could see them and get a feel for them. I also
was interested in the distribution of class examples, so I counted
these, too.

## Model Architecture

**Preprocessing**

As described in the notebook, I augmented the data with the
`perturb.py` source file. Here I rotated the images and did some
zooming to create new examples. This was described in the LeCun
paper. I would have liked to also do translations, but time did not
permit this. I avoided flips because some signs mean different things
when flipped (for instance, left versus right arrows). My overall hope
was to improve generalization on the validation set by creating
synthetic data with transformations that I knew the task was invariant
to.

I normalized the images to grayscale as suggested by LeCun, and then
also centered each image to have mean intensity zero and unit
variance. By normalizing these statistics, the learner's job is
easier. This is done in `dataset.py`.

**Layers**

The network is setup in `network.py`.

I use three convolutionn layers, each with 5x5 kernels. The number of
filters doubles at each layer: 64, 128, 256. I stride by one. I
alternate with max pooling with a stride of two. I then have two fully
connected layers of 1024 and 256 units each.

I use ReLU activations throughout, which allow for more effective
signal backpropagation. I also use batch normalization, which makes it
easier to train deeper networks; this was very helpful. I use dropout
of 50% throughout at each layer; an important choice to get right was
to put the dropout *after* the max pooling layers, since max pooling
resists dropout by pooling information from several units.

**Training**

The typical cross-entropy cost function and AdamOptimizer is setup in
`network.py`. The code that actually does the training is in
`train.py`.

I chose Adam simply as a convenient default; I tried initial learning
rates of 0.1 and 0.01 but those were too high. I got better
performance starting with 0.001. Batch sizes started at 64, which is
small enough to allow a lot of updates to get made in the first epoch,
but large enough to be relatively stable.

I decay the learning rate whenever training cost increases; a failure
to make progress on the training set means either the batch size is
too low (gradient estiamte is too noisy) or the step size is too great
(you start going down hill, and then go back up the other side). I use
a relatively low decay factor (reduce by 20%) because I figure that
decaying is multiplicative and you can rapidly slow down over several
steps. Decaying too fast will make learning too slow too early. This
matters because there is some intrinsic noise in the stochastic
learning cost.

I also increase the batch size every epoch to get more stable
estimates. This matters when you are close to the optimum and the
ratio of noise from the stochastic batching to curvature of the cost
surface gets high.

As a theoretical matter, it is necessary to decay learning rate or
increase batch size to get SGD to converge.

**Training Results**

The validation is accuracy is 97.5%, which I am proud of. The training
accuracy is lower (95.3%), which suggests the model has not overfit,
which means to improve validation performance I think I would have to
add capacity.

Overall, though, I think the most helpful thing would be more training
data. When I experimented with increasing the number of layers or
receptive field size, et cetera, I saw little if any improvement in
validation performance. There are more ways to generate even more
synthetic data, so this could potentially be one easy way to get more
data.

I was disappointed that my attempts to implement the multi-scale
concept that LeCun describes were not very successful. Likewise, I
didn't get improved performance from 1x1 kernels or the inception
architecture. I would like to play more with that in the future, but
perhaps those techniques work better on larger datasets...

I would also have like to experiment with something like adaptive
histogram equalization, since I think the way I adjusted the contrast
made it hard for this human to see what the signs were. I am not sure
I would get 97.5% accuracy myself! That makes the computer's work more
impressive!

## Internet Examples

I acquired 5 examples from the internet. I'm disappointed that my
accuracy was only 60%; top 5 accuracy was also 60%. The model is very
sure of the correct answers, but very confused about the roundabout;
you can see this because multiple top 5 classes have relatively high
probability. It is quite confident in the wrong answer to the wild
animals sign, so I am not sure what happened there.

One thought is that I may have cropped the roundabout sign too
closely.

The test accuracy is 95.3%, so perhaps I made some kind of mistake in
preparing or choosing the internet examples.
