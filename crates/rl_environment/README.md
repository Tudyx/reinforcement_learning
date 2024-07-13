# Environment

Provide the trait to define an reinforcement learning environment. Provide some helper for render and debug them.

Each struct implementing Environment can automatically use as an ArrayEnv or VecEnv.


TODO: explore different environment to detect the constrain:
[x]: basic env
[]: contiguous env 
[]: very big observation env (for instance Atari.)


Add a `from_fn` function for vec and array env? https://doc.rust-lang.org/stable/std/array/fn.from_fn.html