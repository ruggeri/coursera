1. Add kernel_initializer with stddev of 0.02.
2. Could try to bring back conv2d_transpose?
3. Could try to use larger initial image dimension in generator.
4. Try reviewer's suggested sequence of number of filters.
5. Eliminate conv2d's direct dependence on config for NUM_CONV_FILTERS
   and CONV_KSIZE?
