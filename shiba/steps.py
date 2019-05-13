from shiba.utils import repackage_hidden


def default_step(trainer, batch):
    inputs, targets = batch
    inputs = inputs.to(trainer.device, non_blocking=True)
    targets = targets.to(trainer.device, non_blocking=True)
    outputs = trainer.model(inputs)
    loss = trainer.criterion(outputs, targets)
    return dict(loss=loss,
                inputs=inputs,
                outputs=outputs,
                targets=targets)


def rnn_step(trainer, batch):
    hidden = repackage_hidden(trainer.out['hidden'])
    inputs, targets = batch  # inputs.shape : (seq_len, batch_size)
    outputs, hidden = trainer.model(inputs, hidden)
    seq_len, batch_size, vocab_size = outputs.shape
    loss = trainer.criterion(outputs.view(-1, vocab_size), targets.view(-1)) * (seq_len / trainer.seq_len)  # rescale for variable sequence lengths
    return dict(loss=loss,
                inputs=inputs,
                outputs=outputs,
                hidden=hidden,
                targets=targets)
