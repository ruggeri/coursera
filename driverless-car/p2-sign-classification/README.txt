Normalization to gray scale was very important. On the other hand,
adding zoomed and rotated images didn't seem to help that much. One
odd note: my pickled file was much more than 2x the size of the pickle
file I imported. Also: I didn't do any translations because I didn't
see improvement from the other perturbations, and the translation code
would have been more work :-P

Simple series of three 5x5 convolutions each alternated with pooling
layers. Followed by two dense layers and then a final output layer.

It was important to do dropout *after* the pooling. That makes sense
because the impact of dropout will necessarily be lessened by doing
pooling (for instance, all dropout of values not the max is useless).

Adding batch normalization did really seem to help greatly. Also, it
was important to adjust the learning rate down as the epochs
continued, as well as increase the batch size, too. These helped me
advance from 5% to 3% misclassification.

I didn't see a benefit from 1x1 convolutions, even after I started
using batch normalization. I didn't see a real benefit from
concatenating the outputs of the second and third convolution layers
like they did in the LeCun paper. I didn't see any benefit from
replacing my 5x5 convolutions with a greater number of 3x3
convolutions either.

Datafiles live here:

https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
