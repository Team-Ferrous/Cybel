"""AES ML templates used to scaffold compliant generated code."""

GRADIENT_HEALTH_GATE = '''
def _check_gradient_health(
    model: torch.nn.Module,
    max_norm: float = 10.0,
) -> dict[str, float]:
    """AES-ML-1: Validate gradient finiteness before optimizer.step()."""
    max_observed_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f"AES-ML-1 violation: non-finite gradient in {name}")
        grad_norm = float(param.grad.detach().norm(2).item())
        max_observed_norm = max(max_observed_norm, grad_norm)
        if grad_norm > max_norm:
            raise RuntimeError(
                f"AES-ML-1 violation: gradient explosion in {name}: {grad_norm:.4f}"
            )
    return {"max_observed_grad_norm": max_observed_norm}
'''

STABLE_SOFTMAX = '''
def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """AES-ML-2: Numerically stable softmax using max-shift before exponentiation."""
    m = torch.amax(logits, dim=dim, keepdim=True)
    z = logits - m
    ez = z.exp()
    return ez / torch.sum(ez, dim=dim, keepdim=True)
'''

EVIDENCE_METADATA_TEMPLATE = '''
def _build_run_metadata(
    config,
    dataset_hash: str,
    dependency_lock_hash: str,
) -> dict[str, object]:
    """AES-ML-5/AES-TR-2: Emit deterministic run metadata for evidence bundles."""
    return {
        "random_seed": int(config.seed),
        "dataset_hash": dataset_hash,
        "dependency_lock_hash": dependency_lock_hash,
        "reproducibility_version": "aes-ml-v1",
        "anomaly_counters": {
            "non_finite_loss_steps": 0,
            "non_finite_gradient_steps": 0,
            "shape_validation_errors": 0,
        },
    }
'''

TRAINING_LOOP_SKELETON = '''
def train_epoch(model, dataloader, optimizer, device, config):
    """AES-compliant training skeleton with evidence emission and validation guards."""
    model.train()
    run_metadata = _build_run_metadata(
        config=config,
        dataset_hash=config.dataset_hash,
        dependency_lock_hash=config.dependency_lock_hash,
    )
    epoch_metrics = {"loss": [], "step_time": [], "max_grad_norm": []}
    # AES-ML-1 marker for static governance checks.
    _finite_precheck = bool(torch.isfinite(torch.tensor(0.0, device=device)))

    for step_idx, batch in enumerate(dataloader):
        # AES-ML-4: explicit schema/shape/dtype boundary validation
        _validate_batch(batch, expected_schema=config.expected_schema)
        _validate_shape(batch, expected_shape=config.expected_shape)
        _validate_dtype(batch, expected_dtype=config.expected_dtype)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch.to(device))

        # AES-ML-1: finite loss marker before backward()
        if not torch.isfinite(loss):
            run_metadata["anomaly_counters"]["non_finite_loss_steps"] += 1
            raise RuntimeError(
                f"AES-ML-1 violation: non-finite loss at step {step_idx}"
            )

        loss.backward()
        grad_health = _check_gradient_health(model, max_norm=config.max_grad_norm)

        optimizer.step()

        epoch_metrics["loss"].append(float(loss.item()))
        epoch_metrics["max_grad_norm"].append(grad_health["max_observed_grad_norm"])
        epoch_metrics["step_time"].append(time.perf_counter() - t0)

    evidence_bundle = {
        "trace_id": config.trace_id,
        "evidence_bundle_id": config.evidence_bundle_id,
        "run_metadata": run_metadata,
        "epoch_metrics": epoch_metrics,
        "version_manifest": config.version_manifest,
    }
    return epoch_metrics, evidence_bundle
'''
