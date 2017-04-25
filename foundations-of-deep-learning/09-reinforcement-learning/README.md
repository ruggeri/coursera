## No lookahead

I've tried to simplify the task several times. The most basic scenario
is to reward the player constantly. That should be easiest. I use a
discount factor of 0.0, too.

If I reward the player for keeping their paddle aligned with the ball,
then they can learn this strategy perfectly.

It's harder to learn the "ideal" strategy: predict where the ball will
go. One problem was that the computer would move the paddle away at
the last moment. This appeared to be fixed when I stopped giving a
reward when the game reset. I also give more reward to the player when
the ball is nearer to their side; it's less important to act perfectly
when the ball is far away, but it is important when the ball is very
close. With these tweaks, it appears we can learn the ideal strategy.

Weird note: performance got a lot worse after the first epoch of
training!

I got extremely good performance at
d2bf210841114295e194e6d18478c5abdbe5c8e0. This plays basically
perfectly. It took two epochs to train.

But then again, I tried another time and got performance on par with
the hand-built AI. So there may be high variance in training.
