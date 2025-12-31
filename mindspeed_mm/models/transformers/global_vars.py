_FSDP2_EP_RANK = None
_FSDP2_EP_SIZE = None
_FSDP2_CHECK_MOE_FUNC = None


def get_ep_rank():
    global _FSDP2_EP_RANK
    if _FSDP2_EP_RANK:
        return _FSDP2_EP_RANK
    else:
        return 0


def get_ep_size():
    global _FSDP2_EP_SIZE
    if _FSDP2_EP_SIZE:
        return _FSDP2_EP_SIZE
    else:
        return 1


def get_check_moe_func():
    global _FSDP2_CHECK_MOE_FUNC
    if _FSDP2_CHECK_MOE_FUNC:
        return _FSDP2_CHECK_MOE_FUNC
    else:
        return lambda _: False


def set_ep_rank(ep_rank):
    global _FSDP2_EP_RANK
    _FSDP2_EP_RANK = ep_rank


def set_ep_size(ep_size):
    global _FSDP2_EP_SIZE
    _FSDP2_EP_SIZE = ep_size


def set_check_moe_func(fn):
    global _FSDP2_CHECK_MOE_FUNC
    _FSDP2_CHECK_MOE_FUNC = fn