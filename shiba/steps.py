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


def rnn_step(trainer, batch, seq_len=None):
    hidden = repackage_hidden(trainer.out['hidden'])
    inputs, targets = batch  # inputs.shape : (batch, seq)
    inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
    outputs, hidden = trainer.model(inputs, hidden)
    batch_seq_len, batch_size, vocab_size = outputs.shape
    loss = trainer.criterion(outputs.view(-1, vocab_size), targets.view(-1))
    if seq_len:
        loss *= (batch_seq_len / seq_len)  # rescale for variable sequence lengths
    return dict(loss=loss,
                inputs=inputs,
                outputs=outputs,
                hidden=hidden,
                targets=targets)