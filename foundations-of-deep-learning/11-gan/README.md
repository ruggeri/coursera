Works great. Fuck my life, though. The authenticity label was being
silently cast from a float (it's a float because label smoothing) to
an int, which rounds down, so 100% of images were being labeled as
inauthentic. So I lost about a day even though my code really was
working from the beginning.

Oh well.
