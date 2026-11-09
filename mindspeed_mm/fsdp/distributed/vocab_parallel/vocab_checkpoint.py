"""Shard-suffixed checkpoint keys for the host-resident vocab embeddings."""
from .constants import SHARD_KEY_SUFFIX
from .vocab_sparse_grad import get_vocab_embeddings


def get_host_vocab_shard_keys(model) -> dict:
    """``{fqn: suffix}`` of every host-resident vocab embedding of ``model``.

    Each rank holds a different row window of the same embedding, so the key carries the window it
    belongs to -- the reference does the same with ``.ep_shard_00001_of_00002``. Without it every
    rank would write a differently shaped tensor under one name and the checkpoint metadata could
    only keep one of them.
    """
    keys = {}
    for name, embedding in get_vocab_embeddings(model):
        start, end = embedding._vp_vocab_start_index, embedding._vp_vocab_end_index
        window = end - start
        keys[f"{name}.weight"] = SHARD_KEY_SUFFIX.format(start // window, embedding.num_embeddings // window)
    return keys


def add_shard_suffixes(checkpoint_state: dict, keys: dict) -> None:
    """Append each embedding's window suffix to its keys, in place (model or optimizer state dict)."""
    _map_keys(checkpoint_state, keys, add=True)


def remove_shard_suffixes(checkpoint_state: dict, keys: dict) -> None:
    """Undo :func:`add_shard_suffixes` so the dict matches the model's parameter names."""
    _map_keys(checkpoint_state, keys, add=False)


def _map_keys(checkpoint_state: dict, keys: dict, add: bool) -> None:
    if not keys:
        return

    def convert(name: str) -> str:
        # ``get_optimizer_state_dict`` keys the moments by fqn. A flattened dict keeps that but prefixes
        # the key with ``state.``/``param_groups.``, so the prefix is carried aside while the fqn matches.
        prefix, rest = "", name
        for candidate in ("state.", "param_groups."):
            if name.startswith(candidate):
                prefix, rest = candidate, name[len(candidate):]
                break
        for fqn, suffix in keys.items():
            if add and (rest == fqn or rest.startswith(fqn + ".")):
                return prefix + fqn + suffix + rest[len(fqn):]
            if not add and rest.startswith(fqn + suffix):
                return prefix + fqn + rest[len(fqn) + len(suffix):]
        return name

    if "state" not in checkpoint_state:  # a model state dict: fqn -> tensor
        for name in [n for n in checkpoint_state if convert(n) != n]:
            checkpoint_state[convert(name)] = checkpoint_state.pop(name)
        return

    state = checkpoint_state.get("state") or {}
    for name in [n for n in state if convert(n) != n]:
        state[convert(name)] = state.pop(name)
    for group in checkpoint_state.get("param_groups") or ():
        group["params"] = [p if not isinstance(p, str) else convert(p) for p in group["params"]]
