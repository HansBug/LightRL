from types import SimpleNamespace

import pytest
import torch

from slime.world_model.loss_hook import apply_world_model_loss


def _args(**overrides):
    base = dict(
        world_model_enable=True,
        world_model_loss_coef=1.0,
        world_model_loss_hook_path=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_hook_tolerates_partial_target_coverage():
    """A None target entry must not zero the whole micro-batch's aux loss."""
    logits = torch.zeros(1)
    pred = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([2.0, 2.0])]
    target = [torch.tensor([1.0, 0.0]), None, torch.tensor([0.0, 2.0])]
    out_loss, log = apply_world_model_loss(
        args=_args(),
        batch={"wm_pred_latents": pred, "wm_target_latents": target},
        logits=logits,
        loss=logits.sum(),
        reported_loss={},
    )
    # Only rows 0 and 2 contribute: (0 + mean((2,2)-(0,2))^2=2.0) / 2 = 1.0
    assert torch.isclose(out_loss, torch.tensor(1.0))
    assert torch.isclose(log["wm/loss"], torch.tensor(1.0))
    assert torch.isclose(log["wm/target_mask_fraction"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(log["wm/latent_available"], torch.tensor(1.0))


def test_hook_all_targets_missing_is_noop():
    logits = torch.ones(2, 3)
    base_loss = logits.sum()
    pred = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    out_loss, log = apply_world_model_loss(
        args=_args(),
        batch={"wm_pred_latents": pred, "wm_target_latents": [None, None]},
        logits=logits,
        loss=base_loss,
        reported_loss={},
    )
    assert torch.isclose(out_loss, base_loss)
    assert torch.isclose(log["wm/loss"], torch.tensor(0.0))
    assert torch.isclose(log["wm/latent_available"], torch.tensor(0.0))
    assert torch.isclose(log["wm/target_mask_fraction"], torch.tensor(0.0))


def test_hook_presence_mask_combines_with_explicit_mask():
    logits = torch.zeros(1)
    pred = [torch.tensor([1.0, 0.0]), torch.tensor([5.0, 5.0]), torch.tensor([2.0, 0.0])]
    target = [torch.tensor([1.0, 0.0]), torch.tensor([9.0, 9.0]), None]
    out_loss, log = apply_world_model_loss(
        args=_args(),
        batch={
            "wm_pred_latents": pred,
            "wm_target_latents": target,
            # Provider masked out row 1 explicitly; row 2 has no target.
            "wm_target_mask": [1.0, 0.0, 1.0],
        },
        logits=logits,
        loss=logits.sum(),
        reported_loss={},
    )
    # Only row 0 survives both masks: loss 0.0.
    assert torch.isclose(out_loss, torch.tensor(0.0))
    assert torch.isclose(log["wm/target_mask_fraction"], torch.tensor(1.0 / 3.0))


def test_hook_mask_fraction_reports_real_coverage_for_backprop_projection():
    logits = torch.randn(1, 6, 11, requires_grad=True)
    target = [torch.zeros(4), None, torch.zeros(4)]
    _, log = apply_world_model_loss(
        args=_args(
            world_model_backprop_to_llm=True,
            world_model_latent_dim=4,
            world_model_policy_projection="hash",
            qkv_format="thd",
        ),
        batch={
            "wm_target_latents": target,
            "total_lengths": [2, 2, 2],
            "response_lengths": [1, 1, 1],
        },
        logits=logits,
        loss=logits.sum() * 0.0,
        reported_loss={},
    )
    assert torch.isclose(log["wm/target_mask_fraction"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(log["wm/policy_gradient_path"], torch.tensor(1.0))
