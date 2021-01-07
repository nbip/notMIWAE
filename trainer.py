import time
import sys


def train(model, batch_size, max_iter=10000, name=None):

    if name is not None:
        model.save(name)

    start = time.time()
    best = float("inf")

    for i in range(max_iter):
        loss = model.train_batch(batch_size=batch_size)

        if i % 100 == 0:
            took = time.time() - start
            start = time.time()

            val_loss = model.val_batch()

            if val_loss < best and name is not None:
                best = val_loss
                model.save(name)
            print("{0}/{1} updates, {2:.2f} s, {3:.2f} train_loss, {4:.2f} val_loss"
                  .format(i, max_iter, took, loss, val_loss))
            sys.stdout.flush()
