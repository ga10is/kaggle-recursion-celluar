import torch


def save_checkpoint(state, is_best, fpath='checkpoint.pth'):
    torch.save(state, fpath)
    if is_best:
        torch.save(state, 'best_model.pth')


def load_checkpoint(_model, _optimizer, _scheduler, fpath, _metric_fc=None):
    checkpoint = torch.load(fpath)
    # reset optimizer setting
    _epoch = checkpoint['epoch']
    _optimizer.load_state_dict(checkpoint['optimizer'])
    _scheduler.load_state_dict(checkpoint['scheduler'])
    _model.load_state_dict(checkpoint['state_dict'])
    if _metric_fc is not None:
        _metric_fc.load_state_dict(checkpoint['metric_fc'])

    return _epoch, _model, _optimizer, _scheduler, _metric_fc
