import torch
from shiba.utils import repackage_hidden


def default_train_step(batch, core):
    inputs, targets = batch
    inputs = inputs.to(core.device, non_blocking=True)
    targets = targets.to(core.device, non_blocking=True)
    outputs = core.model(inputs)
    loss = core.criterion(outputs, targets)
    return dict(loss=loss,
                inputs=inputs,
                outputs=outputs,
                targets=targets)

# by default, eval step is just the training step with `no_grad`
@torch.no_grad()
def default_eval_step(batch, core):
    return default_train_step(batch, core)


def rnn_step(batch, core):
    """An example RNN stop
    Args:
        batch:
        core:

    Returns:
    """
    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    hidden = repackage_hidden(core.train_out['hidden'])
    inputs, targets = batch  # inputs.shape : (seq_len, batch_size)
    outputs, hidden = core.model(inputs, hidden)
    seq_len, batch_size, vocab_size = outputs.shape
    loss = core.criterion(outputs.view(-1, vocab_size), targets.view(-1)) * (seq_len / core.seq_len)  # rescale for variable sequence lengths
    return dict(loss=loss,
                inputs=inputs,
                outputs=outputs,
                hidden=hidden,
                targets=targets)


@torch.no_grad()
def rnn_eval_step(batch, core):
    return rnn_step(batch, core)
