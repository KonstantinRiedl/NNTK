import torch

from model import compute_loss, flatten, unflatten


def _trainable_entries(v, model):
    parts = []
    offset = 0
    for p in model.parameters():
        n = p.numel()
        if p.requires_grad:
            parts.append(v[offset:offset + n])
        offset += n
    return torch.cat(parts)


def make_hvp(model, x, y):
    # Returns v_dict -> (H_W1, H_b1, H_W2). The v-independent quantities are
    # computed once here, so the closure is only valid while the parameters
    # stay unchanged (i.e. within one Newton step).
    B = x.size(0)
    n = model.hidden_dim
    scaling = 1.0 / (n ** model.beta)

    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W2 = model.fc2.weight.detach()

    z1 = x @ W1.T + b1
    a1 = torch.tanh(z1)
    u = scaling * (a1 @ W2.T)

    g_u = (u - y) / B
    g_tilde_u = scaling * g_u
    tanh_p = 1.0 - a1**2
    tanh_pp = -2.0 * a1 * tanh_p
    g_W2_tanh_pp = (g_tilde_u @ W2) * tanh_pp

    def hvp(v_dict):
        v_W1 = v_dict["fc1.weight"]
        v_b1 = v_dict["fc1.bias"]
        v_W2 = v_dict["fc2.weight"]

        R_z1 = x @ v_W1.T + v_b1
        R_a1 = tanh_p * R_z1
        R_u = scaling * (a1 @ v_W2.T + R_a1 @ W2.T)

        R_g_tilde_u = scaling * (R_u / B)
        R_delta1 = (R_g_tilde_u @ W2 + g_tilde_u @ v_W2) * tanh_p + g_W2_tanh_pp * R_z1

        H_W1 = R_delta1.T @ x
        H_b1 = R_delta1.sum(dim=0)
        H_W2 = R_g_tilde_u.T @ a1 + g_tilde_u.T @ R_a1

        return H_W1, H_b1, H_W2

    return hvp


def make_gnvp(model, x):
    # Returns v_dict -> (GN_W1, GN_b1, GN_W2); same caching caveat as make_hvp.
    B = x.size(0)
    n = model.hidden_dim
    scaling = 1.0 / (n ** model.beta)

    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W2 = model.fc2.weight.detach()

    z1 = x @ W1.T + b1
    a1 = torch.tanh(z1)
    tanh_p = 1.0 - a1**2

    def gnvp(v_dict):
        v_W1 = v_dict["fc1.weight"]
        v_b1 = v_dict["fc1.bias"]
        v_W2 = v_dict["fc2.weight"]

        R_z1 = x @ v_W1.T + v_b1
        R_a1 = tanh_p * R_z1

        R_u = scaling * (a1 @ v_W2.T + R_a1 @ W2.T)

        g_gn = scaling * (R_u / B)
        delta1_gn = (g_gn @ W2) * tanh_p

        GN_W1 = delta1_gn.T @ x
        GN_b1 = delta1_gn.sum(dim=0)
        GN_W2 = g_gn.T @ a1

        return GN_W1, GN_b1, GN_W2

    return gnvp


def conjugate_gradient(HVP_callable, b, adaptive_damping, max_iter=10, tol=1e-4, p0=None, preconditioner=None):
    if p0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = p0.clone()
        r = b - (HVP_callable(x) + adaptive_damping * x)

    z = r if preconditioner is None else preconditioner(r)
    p = z.clone()

    rzold = torch.dot(r, z)
    rz0 = rzold.clone()

    if rzold <= 1e-30:
        return x, 0, "converged"

    for iters in range(1, max_iter + 1):
        Ap = HVP_callable(p) + adaptive_damping * p
        pAp = torch.dot(p, Ap)

        if torch.abs(pAp) < 1e-20:
            return x, iters, "near_zero_curvature"
        if pAp <= 0:
            return None, iters, "negative_curvature"

        alpha = rzold / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        z = r if preconditioner is None else preconditioner(r)
        rznew = torch.dot(r, z)

        if rznew < tol**2 * rz0:
            return x, iters, "converged"

        beta = rznew / rzold
        p = z + beta * p
        rzold = rznew

    return x, max_iter, "max_iter"


def adaptive_step(model, optimizer, step, loss_old, ordered_params, inputs, targets):
    old_params = [p.data.clone() for p in ordered_params]
    optimizer.apply_step(step)

    with torch.no_grad():
        loss_new = compute_loss(model(inputs), targets).item()

    if loss_new < loss_old:
        optimizer.param_groups[0]["damping"] *= 0.8
        accepted = True
    else:
        for p, old_p in zip(ordered_params, old_params):
            p.data.copy_(old_p)
        optimizer.param_groups[0]["damping"] *= 2.0
        accepted = False

    optimizer.param_groups[0]["damping"] = torch.clamp(
        torch.as_tensor(optimizer.param_groups[0]["damping"], dtype=torch.float64),
        1e-8,
        1e2,
    ).item()

    return accepted, loss_new


def diagonal_GN_preconditioner(model, inputs, damping=0.0, eps=0.0):
    B = inputs.size(0)
    N = model.hidden_dim
    scaling = 1.0 / (N ** model.beta)

    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W2 = model.fc2.weight.detach()

    z1 = inputs @ W1.T + b1
    a1 = torch.tanh(z1)
    tanh_p = 1.0 - a1**2
    w2_sq = W2.pow(2).sum(dim=0)

    diag_W2 = scaling**2 * a1.pow(2).mean(dim=0)
    diag_W2 = diag_W2.repeat(W2.shape[0])

    diag_b1 = scaling**2 * w2_sq * tanh_p.pow(2).mean(dim=0)
    diag_W1 = scaling**2 * w2_sq[:, None] * ((tanh_p.pow(2)).T @ inputs.pow(2)) / B

    diag = torch.cat([diag_W1.reshape(-1), diag_b1.reshape(-1), diag_W2.reshape(-1)])
    M_inv = 1.0 / (diag + damping + eps)

    def apply(r):
        return _trainable_entries(M_inv, model) * r

    return apply


def diagonal_Hessian_preconditioner(model, inputs, targets, damping=0.0, eps=0.0):
    B = inputs.size(0)
    N = model.hidden_dim
    scaling = 1.0 / (N ** model.beta)

    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W2 = model.fc2.weight.detach()

    z1 = inputs @ W1.T + b1
    a1 = torch.tanh(z1)

    u = scaling * (a1 @ W2.T)
    residual = (u - targets) / B

    tanh_p = 1.0 - a1**2
    tanh_pp = -2.0 * a1 * tanh_p
    w2_sq = W2.pow(2).sum(dim=0)

    diag_W2 = scaling**2 * a1.pow(2).mean(dim=0)
    diag_W2 = diag_W2.repeat(W2.shape[0])

    diag_b1_GN = scaling**2 * w2_sq * tanh_p.pow(2).mean(dim=0)
    diag_W1_GN = scaling**2 * w2_sq[:, None] * ((tanh_p.pow(2)).T @ inputs.pow(2)) / B

    correction = scaling * (residual @ W2) * tanh_pp
    diag_b1_H = correction.mean(dim=0)
    diag_W1_H = (correction.T @ inputs.pow(2)) / B

    diag_b1 = (diag_b1_GN + diag_b1_H).abs()
    diag_W1 = (diag_W1_GN + diag_W1_H).abs()

    diag = torch.cat([diag_W1.reshape(-1), diag_b1.reshape(-1), diag_W2.reshape(-1)])
    M_inv = 1.0 / (diag + damping + eps)

    def apply(r):
        return _trainable_entries(M_inv, model) * r

    return apply


def neuron_block_preconditioner(model, inputs, damping=0.0, eps=1e-8):
    trainable = [
        name
        for name, p in (
            ("fc1.weight", model.fc1.weight),
            ("fc1.bias", model.fc1.bias),
            ("fc2.weight", model.fc2.weight),
        )
        if p.requires_grad
    ]
    if trainable != ["fc1.weight", "fc1.bias", "fc2.weight"]:
        raise NotImplementedError(
            "neuron_block_preconditioner assumes fc1.weight, fc1.bias and fc2.weight "
            "are all trainable; the current model freezes some of them, which would "
            "misalign the slices in apply()."
        )

    B = inputs.shape[0]
    N = model.hidden_dim
    d = inputs.shape[1]
    scaling = 1.0 / (N ** model.beta)

    W1 = model.fc1.weight.detach()
    b1 = model.fc1.bias.detach()
    W2 = model.fc2.weight.detach()

    z1 = inputs @ W1.T + b1
    tanh_p = 1.0 - torch.tanh(z1) ** 2

    X_aug = torch.cat([inputs, torch.ones(B, 1, device=inputs.device)], dim=1)
    w2_sq = W2.pow(2).sum(dim=0)

    blocks = []
    for j in range(N):
        coeff = scaling**2 * w2_sq[j]
        weighted_X = (tanh_p[:, j:j+1] ** 2) * X_aug
        G_j = coeff * (weighted_X.T @ X_aug) / B
        G_j = G_j + (damping + eps) * torch.eye(d + 1, device=inputs.device)
        blocks.append(G_j)

    def apply(r):
        W1_size = N * d
        b1_size = N

        r_W1 = r[:W1_size].view(N, d)
        r_b1 = r[W1_size:W1_size + b1_size]
        r_W2 = r[W1_size + b1_size:]

        z_W1 = torch.zeros_like(r_W1)
        z_b1 = torch.zeros_like(r_b1)

        for j in range(N):
            r_block = torch.cat([r_W1[j], r_b1[j:j+1]])
            z_block = torch.linalg.solve(blocks[j], r_block)
            z_W1[j] = z_block[:d]
            z_b1[j] = z_block[d]

        z_W2 = r_W2
        return torch.cat([z_W1.reshape(-1), z_b1, z_W2])

    return apply
