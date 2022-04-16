import torch
import torch.nn.functional as F
import numpy as np


def log_softmax(output):
    total = np.sum(np.exp(output), axis=1, keepdims=True)
    return output - np.log(total)


def cross_entropy(predictions, targets, epsilon=1e-12):
    N = predictions.shape[0]
    cee = -np.sum(targets * np.log(predictions + epsilon)) / N
    return cee


def nlllose(log_logit, targets):
    """
    negative log likelihood loss
    """
    targets = np.argmax(targets, axis=1)
    out = np.empty(targets.shape, dtype=np.float32)
    for batch_idx in range(len(targets)):  # for every batch
        out[batch_idx] = log_logit[batch_idx][targets[batch_idx]]

    return -np.mean(out)


def cross_entropy_better(predictions, targets):
    """
    easier to backprop
    """
    log_predictions = log_softmax(predictions)

    return nlllose(log_predictions, targets)


def main():
    raw_outputs = np.array([[0.25, 0.25, 0.25, 0.25],
                            [0.01, 0.01, 0.01, 0.96]])
    targets = np.array([[0, 0, 0, 1],
                        [0, 0, 0, 1]], dtype=np.float32)

    log_softmax_imperatively = log_softmax(raw_outputs)
    log_softmax_torch = F.log_softmax(torch.from_numpy(raw_outputs), dim=1)
    print('======== log_softmax ========')
    print(f'expected: {log_softmax_torch}\ncalculated: {log_softmax_imperatively}', end="\n\n\n")

    nll_torch = F.nll_loss(torch.from_numpy(log_softmax_imperatively), torch.from_numpy(np.argmax(targets, axis=1)))
    nll_imperative = nlllose(log_softmax_imperatively, targets)
    print('======== nll_loss ========')
    print(f'expected: {nll_torch}\ncalculated: {nll_imperative}', end="\n\n\n")

    cee_imperative1 = cross_entropy(raw_outputs, targets)
    cee_imperative2 = cross_entropy_better(raw_outputs, targets)
    cee_torch = F.cross_entropy(torch.from_numpy(raw_outputs), torch.from_numpy(targets))
    print(cee_imperative1, cee_imperative2, cee_torch)
    print('======== cross entropy loss ========')
    print(f'classic cee: {cee_imperative1}')
    print(f'expected: {cee_torch}\ncalculated: {cee_imperative2}', end="\n\n\n")


if __name__ == '__main__':
    main()
