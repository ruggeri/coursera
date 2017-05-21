Suggested configuration:

```
Bring back conv2d_transpose? Use strides of 2 and replace maxpooling.
NUM_DISCRIMINATOR_FILTERS = [64, 256, 512]
GENERATOR_INITIAL_SHAPE = (7, 7, 512)
Use bn_relu instead of tanh for initial image.
NUM_GENERATOR_FILTERS = [256, 128, 64, out_channel_dim]
NUM_GENERATOR_STRIDES = [  2,   2,  1, 1]
```
