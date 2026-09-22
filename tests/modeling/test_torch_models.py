import torch

from maritime_cbm.modeling.torch_models import (
    LinearResidualMLPRegressor,
    build_torch_candidate_specs,
    build_torch_model,
    count_trainable_parameters,
    torch_candidate_from_dict,
)


def test_candidate_grid_has_six_frozen_specs_and_expected_parameter_counts() -> None:
    specs = build_torch_candidate_specs()

    assert len(specs) == 6
    assert len({spec.candidate_id for spec in specs}) == 6
    counts = {
        spec.candidate_id: count_trainable_parameters(build_torch_model(spec)) for spec in specs
    }
    assert counts == {
        "mlp_raw_hidden_64_32": 2_978,
        "mlp_raw_hidden_128_64": 10_050,
        "mlp_speed_centered_hidden_64_32": 2_978,
        "mlp_speed_centered_hidden_128_64": 10_050,
        "linear_residual_mlp_speed_centered_hidden_64_32": 3_004,
        "linear_residual_mlp_speed_centered_hidden_128_64": 10_076,
    }


def test_all_candidates_return_two_unclipped_outputs() -> None:
    features = torch.randn(5, 12)

    for spec in build_torch_candidate_specs():
        predictions = build_torch_model(spec)(features)

        assert predictions.shape == (5, 2)


def test_linear_residual_model_starts_at_linear_path_and_learns_residual() -> None:
    torch.manual_seed(42)
    spec = next(
        spec
        for spec in build_torch_candidate_specs()
        if spec.candidate_id == "linear_residual_mlp_speed_centered_hidden_64_32"
    )
    model = build_torch_model(spec)
    assert isinstance(model, LinearResidualMLPRegressor)
    features = torch.randn(8, 12)

    with torch.no_grad():
        initial_difference = model(features) - model.linear_path(features)
    assert torch.equal(initial_difference, torch.zeros_like(initial_difference))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.mse_loss(model(features), torch.ones(8, 2))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        learned_difference = model(features) - model.linear_path(features)
    assert torch.count_nonzero(learned_difference) > 0


def test_candidate_payload_must_match_frozen_grid() -> None:
    spec = build_torch_candidate_specs()[0]
    payload = spec.to_dict()

    assert torch_candidate_from_dict(payload) == spec

    payload["hidden_sizes"] = [999, 999]
    try:
        torch_candidate_from_dict(payload)
    except ValueError as error:
        assert "approved grid" in str(error)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("Modified candidate payload should be rejected")
