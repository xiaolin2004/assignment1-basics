from __future__ import annotations
import math

def lr_cosine_schedule(it,
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters):
    if it  < warmup_iters:
        return it/warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        return min_learning_rate+(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))/2*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
