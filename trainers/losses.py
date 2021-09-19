import torch
import torch.nn.functional as F
import math
import numpy as np

from utils.utils import EPSILON

__all__ = ["mse_loss", "mult_ce_loss"]

bce_loss = torch.nn.BCELoss(reduction='none')

PI = 3.1415926


def mse_loss(data, logits, weight):
    """Mean square error loss."""
    weights = torch.ones_like(data)
    weights[data > 0] = weight
    res = weights * (data - logits)**2
    return res.sum(1)


def is_converge(new_para, para, rtol=1e-05, atol=1e-08):
    return torch.allclose(new_para, para, rtol=rtol, atol=atol)


def EM_opt(data, logits, sigma, bar_r):
    bar_sigma = sigma
    num_u, num_i = data.shape
    gamma_u = torch.full((num_u, 1), 0.8, dtype=float).cuda()
    iota_i = torch.full((1, num_i), 0.8, dtype=float).cuda()
    eta_u = torch.full((num_u, 1), 0.8, dtype=float).cuda()
    zeta_i = torch.full((1, num_i), 0.8, dtype=float).cuda()

    hat_normal_0_exp = torch.exp(-torch.div(logits**2, 2 * sigma**2))
    hat_normal_PDF_0 = torch.div(hat_normal_0_exp, (math.sqrt(2 * PI) * sigma))
    bar_normal_0_exp = torch.exp(-torch.div(bar_r**2, 2 * bar_sigma**2))
    bar_normal_PDF_0 = torch.div(bar_normal_0_exp,
                                 (math.sqrt(2 * PI) * bar_sigma))

    hat_normal_1_exp = torch.exp(-torch.div((logits - 1)**2, 2 * sigma**2))
    hat_normal_PDF_1 = torch.div(hat_normal_1_exp, (math.sqrt(2 * PI) * sigma))
    bar_normal_1_exp = torch.exp(-torch.div((bar_r - 1)**2, 2 * bar_sigma**2))
    bar_normal_PDF_1 = torch.div(bar_normal_1_exp,
                                 (math.sqrt(2 * PI) * bar_sigma))

    delta_p_ui = None
    psi_p_ui = None
    while True:
        delta_eq_1 = torch.div(
            gamma_u.repeat(1, num_i) + iota_i.repeat(num_u, 1), 2.0)
        psi_eq_1 = torch.div(
            eta_u.repeat(1, num_i) + zeta_i.repeat(num_u, 1), 2.0)

        delta_exp_r0 = torch.mul(psi_eq_1, hat_normal_PDF_0) + torch.mul(
            (1 - psi_eq_1), bar_normal_PDF_0)

        delta_numerator = torch.mul(delta_eq_1, delta_exp_r0)
        delta_denominator = delta_numerator + (1 - delta_eq_1)
        delta_p_ui = torch.div(delta_numerator, delta_denominator)
        delta_p_ui[data > 0] = 1.0

        delta_p_ui[delta_p_ui >= 0.5] = 1
        delta_p_ui[delta_p_ui < 0.5] = 0

        psi_exp_delta1_r0 = torch.div(torch.mul(psi_eq_1, hat_normal_PDF_0),
                                      delta_exp_r0)

        delta_exp_r1 = torch.mul(psi_eq_1, hat_normal_PDF_1) + torch.mul(
            (1 - psi_eq_1), bar_normal_PDF_1)
        psi_exp_delta1_r1 = torch.div(torch.mul(psi_eq_1, hat_normal_PDF_1),
                                      delta_exp_r1)
        psi_p_ui = torch.where(data > 0, psi_exp_delta1_r1, psi_exp_delta1_r0)
        psi_p_ui[delta_p_ui < 0.5] = np.nan

        new_gamma = torch.mean(delta_p_ui, dim=1).reshape(-1, 1)
        new_iota = torch.mean(delta_p_ui, dim=0).reshape(1, -1)

        new_eta = torch.from_numpy(np.nanmean(psi_p_ui.cpu().numpy(),
                                              axis=1)).reshape(-1, 1).cuda()
        new_zeta = torch.from_numpy(np.nanmean(psi_p_ui.cpu().numpy(),
                                               axis=0)).reshape(1, -1).cuda()

        if is_converge(new_gamma, gamma_u) and is_converge(
                new_iota, iota_i) and is_converge(
                    new_eta, eta_u) and is_converge(new_zeta, zeta_i):
            gamma_u = new_gamma
            iota_i = new_iota
            eta_u = new_eta
            zeta_i = new_zeta
            psi_p_ui[psi_p_ui >= 0.5] = 1
            psi_p_ui[psi_p_ui < 0.5] = 0
            break
        gamma_u = new_gamma
        iota_i = new_iota
        eta_u = new_eta
        zeta_i = new_zeta
    return gamma_u, iota_i, eta_u, zeta_i, delta_p_ui, psi_p_ui


def pgm_loss(data, logits, weight, sigma, bar_r=0.1):
    gamma_u, iota_i, eta_u, zeta_i, delta_p_ui, psi_p_ui = EM_opt(
        data,
        logits.clone().detach(), sigma, bar_r)

    delta_1_psi_1 = torch.mul(delta_p_ui, psi_p_ui)
    delta_1_psi_1[torch.isnan(delta_1_psi_1)] = 0
    mask_ratio_delta_1_psi_1 = float(
        torch.sum(delta_1_psi_1).item() /
        (delta_1_psi_1.shape[0] * delta_1_psi_1.shape[1]))

    weights = torch.ones_like(data)
    weights[data > 0] = weight
    mse = weights * (data - logits)**2
    res = delta_1_psi_1 * mse / mask_ratio_delta_1_psi_1

    return res.sum(1)


def WMW_loss(logits_topK, logits_target_items, offset=0.01):
    loss = 0
    for i in range(logits_target_items.shape[1]):
        cur_target_logits = logits_target_items[:, i].reshape(-1, 1)
        x = logits_topK - cur_target_logits
        x = -1.0 * x / offset
        g = 1.0 / (1 + torch.exp(x))
        loss += torch.sum(g)
    return loss


def WMW_loss_sigmoid(logits_topK, logits_target_items, offset=0.01):
    loss = 0
    for i in range(logits_target_items.shape[1]):
        cur_target_logits = logits_target_items[:, i].reshape(-1, 1)
        x = (logits_topK - cur_target_logits) / offset
        g = torch.sigmoid(x)
        loss += torch.sum(g)
    return loss


def mult_ce_loss(data, logits):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs * data

    instance_data = data.sum(1)
    instance_loss = loss.sum(1)
    res = instance_loss / (instance_data + EPSILON)
    return res
