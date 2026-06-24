from typing import Dict, List
import torch
import torch.distributed as dist


class DupExpertExecutor:
    """
    Executor class for handling duplicated experts communication in distributed training.
    Manages parameter and gradient synchronization across different processes for duplicated experts.
    """
    def __init__(self, ep_group, num_experts, max_dup_experts_num):
        """
        Initialize the DupExpertExecutor with group information and expert configuration.

        Args:
            ep_group: Expert parallel process group
            num_experts: Total number of experts
            max_dup_experts_num: Maximum number of duplicated experts per process
        """
        self.num_dup_experts = max_dup_experts_num
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_group_id = dist.get_rank() // dist.get_world_size(ep_group)
        self.num_local_experts = num_experts // dist.get_world_size(ep_group)
        self.wait_works = {}

        # Dictionary to store gradients of duplicated experts
        self.dup_experts_grad = {}

    def async_experts_param_comm(self, dup_experts_map, local_experts, name):
        """
        Asynchronously communicate expert parameters across processes.

        Args:
            dup_experts_map: Mapping of duplicated experts
            local_experts: Local expert parameters to send
            name: Identifier for this communication operation

        Returns:
            Duplicated expert parameters received from other processes
        """
        send_recv_plans = self.build_param_send_recv_plan(dup_experts_map) # send_plan: {dst_rank_id: [local_expert_ids]}, recv_plan: {src_rank_id: [local_expert_ids]}
        dup_experts, recv_works = self.send_recv_duplicate_experts_param_async(
            send_recv_plans=send_recv_plans,
            local_experts_param=local_experts,
            num_dup_experts=self.num_dup_experts
        )

        self.wait_works[name] = recv_works
        return dup_experts

    def async_experts_grad_comm(self, dup_experts_map, dup_experts_grad, name):
        """
        Asynchronously communicate expert gradients across processes.

        Args:
            dup_experts_map: Mapping of duplicated experts
            dup_experts_grad: Gradients of duplicated experts
            name: Identifier for this communication operation
        """
        param_send_recv_plans = self.build_param_send_recv_plan(dup_experts_map)

        dup_experts_grad_dict, recv_works = self.send_recv_duplicate_experts_grad_async(
            param_send_recv_plans=param_send_recv_plans,
            dup_experts_grad=dup_experts_grad
        )

        self.dup_experts_grad[name] = dup_experts_grad_dict
        self.wait_works[name] = recv_works

    def build_param_send_recv_plan(self, dup_experts_map):
        """
        Build a plan for sending and receiving expert parameters based on the duplication map.

        Args:
            dup_experts_map: Mapping of duplicated experts [ep_size, num_dup_experts]

        Returns:
            List of dictionaries containing send/receive plans with source/destination ranks and IDs
        """
        # dup_experts_map [ep_size, num_dup_experts]
        send_recv_plans = []
        ep_size = dist.get_world_size(self.ep_group)

        for dst_rank in range(ep_size):
            for dup_id, src_expert_id in enumerate(dup_experts_map[dst_rank]):
                if src_expert_id == -1:
                    break

                src_rank = src_expert_id // self.num_local_experts
                src_local_id = src_expert_id % self.num_local_experts
                global_src_rank = self.ep_group_id * ep_size + src_rank
                global_dst_rank = self.ep_group_id * ep_size + dst_rank

                send_recv_plans.append({
                    "src_local_id": src_local_id,
                    "src_rank": global_src_rank,
                    "dst_rank": global_dst_rank,
                    "dup_id": dup_id
                })

        return send_recv_plans

    def send_recv_duplicate_experts_param_async(self, send_recv_plans, local_experts_param, num_dup_experts):
        """
        Asynchronously send and receive duplicated expert parameters.

        Args:
            send_recv_plans: Plans for sending and receiving
            local_experts_param: Local expert parameters
            num_dup_experts: Number of duplicated experts

        Returns:
            Tuple of duplicated experts tensor and list of receive works
        """
        recv_works: List[dist.Work] = []
        input_dim, output_dim = local_experts_param.shape[-2], local_experts_param.shape[-1]
        dup_experts = torch.empty((num_dup_experts, input_dim, output_dim), device=local_experts_param.device, dtype=local_experts_param.dtype)

        for send_recv_plan in send_recv_plans:
            src_local_id, src_rank, dst_rank, dup_id = send_recv_plan["src_local_id"], send_recv_plan["src_rank"], send_recv_plan["dst_rank"], send_recv_plan["dup_id"]
            if src_rank == dist.get_rank():
                dist.isend(local_experts_param[src_local_id:src_local_id+1], dst=dst_rank, tag=src_local_id)
            if dst_rank == dist.get_rank():
                recv_work = dist.irecv(dup_experts[dup_id:dup_id+1], src=src_rank, tag=src_local_id)
                recv_works.append(recv_work)
        return dup_experts, recv_works

    def send_recv_duplicate_experts_grad_async(self, param_send_recv_plans, dup_experts_grad):
        """
        Asynchronously send and receive duplicated expert gradients.
        Note: The send and receive relationship for gradients is opposite to that of parameters.

        Args:
            param_send_recv_plans: Send/receive plans built from parameter communication
            dup_experts_grad: Gradients of duplicated experts

        Returns:
            Tuple of gradient dictionary and list of receive works
        """
        recv_works: List[dist.Work] = []
        input_dim, output_dim = dup_experts_grad.shape[-2], dup_experts_grad.shape[-1]
        dup_experts_grad_dict: Dict = {}

        for param_send_recv_plan in param_send_recv_plans:
            # 梯度的send和recv关系和param是反的
            src_local_id, dst_rank, src_rank, dup_id = param_send_recv_plan["src_local_id"], param_send_recv_plan["src_rank"], param_send_recv_plan["dst_rank"], param_send_recv_plan["dup_id"]
            if src_rank == dist.get_rank():
                dist.isend(dup_experts_grad[dup_id:dup_id+1], dst=dst_rank, tag=src_local_id)
            if dst_rank == dist.get_rank():
                dup_experts_grad_dict[src_local_id] = torch.empty((1, input_dim, output_dim), device=dup_experts_grad.device, dtype=dup_experts_grad.dtype)
                recv_work = dist.irecv(dup_experts_grad_dict[src_local_id], src=src_rank, tag=src_local_id)
                recv_works.append(recv_work)
        return dup_experts_grad_dict, recv_works

    def wait_async_works_finished(self, name):
        """
        Wait for all asynchronous communication operations to finish for a given identifier.

        Args:
            name: Identifier for the communication operations to wait for
        """
        for work in self.wait_works[name]:
            work.wait()
        self.wait_works[name].clear()

    def add_dup_experts_grad(self, local_experts_grad, name):
        """
        Add gradients of duplicated experts to local expert gradients after waiting for async operations.

        Args:
            local_experts_grad: Local expert gradients
            name: Identifier for the duplicated expert gradients to use

        Returns:
            Updated local expert gradients with added duplicated expert gradients
        """
        self.wait_async_works_finished(name)
        local_experts_grad_shape = local_experts_grad.shape
        local_experts_grad = local_experts_grad.view(self.num_local_experts, -1, local_experts_grad.shape[-1])
        for src_local_id, grad in self.dup_experts_grad[name].items():
            local_experts_grad[src_local_id:src_local_id+1] += grad

        local_experts_grad = local_experts_grad.view(local_experts_grad_shape)
        self.dup_experts_grad[name] = {} # free
        return local_experts_grad

    def register_backward_dup_experts_grad_comm_hook(self, dup_experts_map, expert_tensor, name):
        """
        Register a backward hook to communicate gradients of duplicated experts during backpropagation.

        Args:
            dup_experts_map: Mapping of duplicated experts
            expert_tensor: Tensor representing experts that requires gradient communication
            name: Identifier for this communication operation
        """
        def hook_fn(grad):
            with torch.no_grad():
                self.async_experts_grad_comm(
                    dup_experts_map=dup_experts_map,
                    dup_experts_grad=grad[self.num_local_experts:],
                    name=name
                )
            return grad

        if expert_tensor.requires_grad:
            expert_tensor.register_hook(hook_fn)

    def register_backward_dup_experts_grad_acc_hook(self, hook_tensor, name):
        """
        Register a backward hook to accumulate gradients of duplicated experts into local expert gradients.

        Args:
            hook_tensor: Tensor that will have its gradients accumulated
            name: Identifier for the duplicated expert gradients to use
        """
        def hook_fn(grad):
            grad = self.add_dup_experts_grad(grad, name)
            return grad
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(hook_fn)
