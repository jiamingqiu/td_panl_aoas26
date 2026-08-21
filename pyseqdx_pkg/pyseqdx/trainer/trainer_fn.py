import torch.nn as nn

def step_params(x, y, loss_fn, optimizer, clip_max_norm=5.0, model=None):
    # a general wrapper for stepping parameters, usually from 1 batch
    # x, y usually are [batch_size, num_t, input_size], [batch_size, 1]
    # and y are binary
    # the model is passed to loss_fn just in case, if loss_fn is defined
    # on top of the model then it is unnecessary

    optimizer.zero_grad()
    # compute loss
    loss = loss_fn(x=x, y=y, model=model)
    # Backpropagation
    loss.backward()
    # grad clip
    params = [
        p
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    nn.utils.clip_grad_norm_(params, max_norm=clip_max_norm)
    # step
    optimizer.step()

    return loss.item()
