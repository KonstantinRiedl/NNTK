import torch

from model import flatten


def empirical_ntk_gram(model, x):
    # The NTK is taken with respect to the parameters that are actually
    # trainable.  When fc1 is trainable it contributes to the kernel; in the
    # random-feature setting fc1 is frozen and only fc2 contributes.
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Cannot compute an NTK for a model with no trainable parameters")
    grads = []

    for i in range(x.size(0)):
        fi = model(x[i:i+1]).squeeze()
        gi = torch.autograd.grad(fi, params, retain_graph=True, create_graph=False)
        grads.append(flatten([g.detach() for g in gi]))

    J = torch.stack(grads, dim=0)
    K = J @ J.T
    return 0.5 * (K + K.T)


def ntk_eigensystem(model, x):
    K = empirical_ntk_gram(model, x)
    evals, evecs = torch.linalg.eigh(K)
    idx = torch.argsort(evals, descending=True)
    return evals[idx], evecs[:, idx]


def project_onto_basis(fx, basis):
    fx = fx.squeeze(-1)
    return torch.linalg.lstsq(basis, fx).solution.squeeze()


def project_onto_ntk_modes(fx, ntk_evecs):
    fx = fx.squeeze(-1)
    return ntk_evecs.T @ fx
