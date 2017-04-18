I did this on my own and not just through the Jupyter notebook only
because I had a lot of trouble with a bug.

That got me to use TensorBoard, which was helpful. The graph view
showed me that the result from `tf.equal` had suspicious dimension,
which was the cuase of my problem. A simple reshape fixed things. But
it took me hours before I used that graph view...

The accuracy of this model on the validation data after one epoch is
~70%. That's inferior to the simple FFNN we built long ago, so this is
silly. I feel like it should probably be better (so that it can
understand negations), but maybe that needs more units, certainly more
training, etc, and I'm not that interested in this task.
