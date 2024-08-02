# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim_expert_parallel
from megatron.core.transformer.moe.moe_utils import moe_gather, moe_scatter, permute, unpermute
from megatron.core.transformer.transformer_config import TransformerConfig


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config

    @abstractmethod
    def token_permutation(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            indices (torch.Tensor): indices tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self,
        expert_output: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            probs (torch.Tensor): Each token's score with each expert.
            indices (torch.Tensor): The indices used to reorder the expert output.

        Returns:
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
    ) -> None:
        """
        Initialize the zero token dropping router.
        """
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs = None

        # self.indices: The indices of `local_indices` (which holds the un-sorted expert indices of tokens that local expert can process) that give its sorted order along dim 0.
        self.indices = None

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where each element is True if it's between the local_expert_indices. Only useful when cross device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: input tokens of shape [SeqLen/TP, MBS, HiddenSize]
            max_prob: probs of local token assignment to global experts.
            max_ind: token assignment to local experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Permute the tokens across the expert parallel devices.
        if (self.config.tensor_model_parallel_size > 1) or (
            self.config.expert_model_parallel_size > 1
        ):
            with torch.no_grad():
                global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                    max_ind
                )
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                local_indices = global_indices.masked_select(global_local_mask)

            if self.router_topk > 1:  # k > 1
                global_probs = tensor_parallel.gather_from_sequence_parallel_region_to_moe(max_prob)
                self.local_probs = global_probs.masked_select(global_local_mask)
            else:
                self.local_probs = max_prob

            # [S*B/TP, H] -> [S*B, H]
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states, use_global_buffer=True
            )
            # Reshape global_local_mask to be compatible with Tensor.gather
            global_local_map = global_local_mask.nonzero()[:, 0]
            self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
        else:
            if self.router_topk > 1:
                global_local_mask = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_mask)
                self.local_probs = max_prob.masked_select(global_local_mask)
                global_local_map = global_local_mask.nonzero()[:, 0]
                self.global_local_map = global_local_map.view(-1, 1).expand(
                    -1, hidden_states.shape[-1]
                )
                local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
            else:
                local_indices = max_ind
                self.local_probs = max_prob
                local_hidden_states = hidden_states
                self.global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            self.indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather
        self.indices = self.indices.view(-1, 1).expand(-1, hidden_states.shape[-1])
        if self.num_local_experts > 1:
            permuted_local_hidden_states = moe_gather.apply(local_hidden_states, self.indices)
        else:
            permuted_local_hidden_states = local_hidden_states
        return (
            permuted_local_hidden_states,
            tokens_per_expert,
        )

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
    ):
        """
        Reverse process of `dispatch()` which permutes the ouput of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor of shape [sum_tokens_of_all_local_experts, HiddenSize],
            ouput of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [SeqLen/TP, MBS, HiddenSize]
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        scores = self.local_probs.to(dtype=hidden_states.dtype)
        if self.num_local_experts > 1:
            assert self.indices.shape == hidden_states.shape
            unpermuted_local_hidden = moe_scatter.apply(hidden_states, self.indices)
        else:
            unpermuted_local_hidden = hidden_states

        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
        if self.router_topk > 1:
            unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

        unpermuted_local_bias = None
        if self.add_bias:
            assert bias is not None
            unpermuted_local_bias = torch.zeros_like(hidden_states)
            assert self.indices.shape == bias.shape
            unpermuted_local_bias = unpermuted_local_bias.scatter(0, self.indices, bias)
            if self.router_topk > 1:
                unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        if (self.config.tensor_model_parallel_size > 1) or (
            self.config.expert_model_parallel_size > 1
        ):
            assert (
                self.global_local_map is not None
            ), "global_local_map is necessary for `AllGather`."
            ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
            # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = moe_scatter.apply(
                unpermuted_local_hidden, self.global_local_map, global_hidden_shape
            )
            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                unpermuted_global_hidden
            )
            if self.add_bias:
                # Unpermute the bias across expert parallel devices.
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                unpermuted_global_bias = unpermuted_global_bias.scatter_add(
                    0, self.global_local_map, unpermuted_local_bias
                )
                output_bias_total = (
                    tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                        unpermuted_global_bias
                    )
                )
                # bias is duplicated across tensor parallelism ranks;
                # reduce scatter reduces bias across tensor parallel_ranks
                output_bias_total = (
                    output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
                )
        else:
            if self.router_topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )
                if self.add_bias:
                    unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, self.global_local_map, unpermuted_local_bias
                    )

        if self.router_topk == 1:
            output_total = output_total * scores
        output_total = output_total.view(self.hidden_shape)
        if self.add_bias:
            assert output_bias_total is not None
            if self.router_topk == 1:
                output_bias_total = output_bias_total * scores
            output_bias_total = output_bias_total.view(self.hidden_shape)
        else:
            output_bias_total = None

        return output_total, output_bias_total


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll Based Token dispatcher.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config)
        self.hidden_shape = None
        self.num_input_tokens = None
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear
        self.ep_size = config.expert_model_parallel_size
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

        # Token drop and padding.
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called in self.preprocess().
        # The synchronization happens at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = torch.histc(
            indices, bins=self.num_experts, min=0, max=self.num_experts
        )
        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.config.expert_model_parallel_size
        if self.drop_and_pad:
            # probs: [num_experts, capacity]
            self.capacity = self.probs.size(1)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.config.moe_expert_capacity_factor is not None:
            # Token drop but no pad. A synchronization is needed before the first
            # permutation to get the `num_out_tokens` CPU value.
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(
                torch.device("cpu"), non_blocking=True
            )
            self.cuda_sync_point = "before_permutation_1"
        elif ep_size > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.cuda_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed before the token_permutation()
            # function returns to get the `tokens_per_expert` CPU value.
            self.cuda_sync_point = "before_finish"

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert
            ).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ]
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
                torch.device("cpu"), non_blocking=True
            )
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(indices)

        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            indices,
            num_out_tokens=self.num_out_tokens,
            padded_mode=self.drop_and_pad,
        )

        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        global_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                    global_input_tokens, self.global_input_tokens_local_experts_indices
                )
            else:
                global_input_tokens = global_input_tokens.reshape(
                    self.ep_size, self.num_local_experts, self.capacity, -1
                )
                global_input_tokens = (
                    global_input_tokens.transpose(0, 1)
                    .reshape(self.num_local_experts * self.ep_size * self.capacity, -1)
                    .contiguous()
                )

        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )
        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(
                hidden_states
            )

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                hidden_states = unpermute(
                    hidden_states,
                    self.reversed_global_input_permutation_mapping,
                )
            else:
                hidden_states = hidden_states.reshape(
                    self.num_local_experts, self.ep_size, self.capacity, -1
                )
                hidden_states = (
                    hidden_states.transpose(0, 1)
                    .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                    .contiguous()
                )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            hidden_states,
            self.input_splits,
            self.output_splits,
        )

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=self.probs,
            padded_mode=self.drop_and_pad,
            restore_shape=self.hiddden_shape_before_permute,
        )

        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None

class compute_center(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dispatched_input, buckets, E, M, C, n_hash):
        ctx.E = E
        ctx.M = M
        ctx.C = C
        n_bucket = 2
        #dispatched_buckets = buckets.squeeze(0).permute(1,0)
        buckets_chunks = buckets.chunk(E, dim=0)
        input_chunks = dispatched_input.chunk(E, dim=0)
        input_split = torch.empty([E], device=dispatched_input.device, dtype=torch.int64)
        i = 0
        expert_indices = []
        input_list = []
        t_list = []
        diffs = []
        pdist = torch.nn.PairwiseDistance(p=2)
        ou_all = 0.0
        for chunk, input_chunk in zip(buckets_chunks, input_chunks):
            d, s, t = torch.unique(chunk, return_inverse=True, return_counts=True, dim=0)
            
            centers = torch.zeros([d.size(0), M], dtype=dispatched_input.dtype, device=dispatched_input.device, requires_grad=True)
            #input_chunk_reverse = torch.flip(input_chunk, dims=[0])
            #s_reverse = torch.flip(s.flatten(), dims=[0])
            #new_centers = torch.index_copy(centers, 0, s_reverse, input_chunk_reverse)
            #new_centers = torch.index_copy(centers, 0, s.flatten(), input_chunk)
            new_centers = torch.div(torch.index_add(centers, 0, s.flatten(), input_chunk),t.to(dispatched_input.dtype)[:,None].expand(-1,M))
            t_list.append(t)
            
            input_list.append(new_centers)
            expert_indices.append(s.flatten().long())
            diffs.append(input_chunk - torch.index_select(new_centers, 0, s.flatten().long()))
            input_split[i] = d.size(0)
            i += 1
        expert_indices = torch.cat(expert_indices, dim=0)
        t_list = torch.cat(t_list, dim=0)
        diffs = torch.cat(diffs, dim=0)
        ctx.save_for_backward(expert_indices, t_list, input_split)
        dispatched_input = torch.cat(input_list)
        ou = 0
        # return ou, dispatched_input, input_split, torch.cat(expert_indices,dim=0), torch.cat(t_list,dim=0), torch.cat(diffs,dim=0))
        return ou, dispatched_input, input_split, expert_indices, t_list, diffs
        # return dispatched_input

    @staticmethod
    def backward(ctx, ou, grad_output, input_split, expert_indices_1, t_list, diffs):
        E = ctx.E
        M = ctx.M
        C = ctx.C
        expert_indices, t, input_split = ctx.saved_tensors
        new_input_split = torch.zeros_like(input_split)
        input_split = torch.cumsum(input_split, dim=0)
        new_input_split[1:] = input_split[:-1]
        new_input_split = new_input_split.repeat_interleave(C)
        new_expert_indices = expert_indices + new_input_split
        new_grad_output = torch.index_select(torch.div(grad_output, t.float()[:, None].expand(-1, M)), 0, new_expert_indices.flatten())
        return new_grad_output, None, None, None, None
        #return grad_output, None, None, None, None

class recover_token(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, input_split, expert_indices, t_list, C):
        S, M = expert_output.size()
        ctx.S = S
        ctx.M = M
        new_input_split = torch.zeros_like(input_split)
        input_split = torch.cumsum(input_split, dim=0)
        new_input_split[1:] = input_split[:-1]
        new_expert_indices = expert_indices + new_input_split.repeat_interleave(C)
        ctx.device = expert_output.device
        expert_output = torch.index_select(expert_output, 0, new_expert_indices)
        ctx.save_for_backward(t_list, new_expert_indices)
        return expert_output

    @staticmethod
    def backward(ctx, grad_output):
        t_list, new_expert_indices = ctx.saved_tensors
        grad = torch.zeros(size=(ctx.S, ctx.M), device=ctx.device)
        grad_now = torch.index_add(grad, 0, new_expert_indices, grad_output)
        #grad_now = torch.index_copy(grad, 0, new_expert_indices, grad_output) * t_list.float()[:,None].expand(-1,ctx.M)
        return grad_now, None, None, None, None
        #return grad_output, None, None, None, None

compute_center_fn = compute_center.apply
recover_token_fn = recover_token.apply 

class LSHMoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll Based Token dispatcher.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
        """
        super().__init__(config=config)
        self.hidden_shape = None
        self.num_input_tokens = None
        self.num_local_experts = num_local_experts
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continous"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear
        self.ep_size = config.expert_model_parallel_size
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

        # Token drop and padding.
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation() in some cases,
        # because there are several non-blocking DtoH data transfers called in self.preprocess().
        # The synchronization happens at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Preprocess token indices for AlltoAll communication and token permutation. This method computes the number of tokens assigned to each expert based on the input indices.
        It also initializes the necessary data structures for AlltoAll communication, such as input
        and output splits, and the mapping between global tokens and local experts.

        Args:
            indices (torch.Tensor): Tensor of indices mapping tokens to experts.

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = torch.histc(
            indices, bins=self.num_experts, min=0, max=self.num_experts
        )
        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.config.expert_model_parallel_size
        if self.drop_and_pad:
            # probs: [num_experts, capacity]
            self.capacity = self.probs.size(1)
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.config.moe_expert_capacity_factor is not None:
            # Token drop but no pad. A synchronization is needed before the first
            # permutation to get the `num_out_tokens` CPU value.
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(
                torch.device("cpu"), non_blocking=True
            )
            self.cuda_sync_point = "before_permutation_1"
        elif ep_size > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.cuda_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed before the token_permutation()
            # function returns to get the `tokens_per_expert` CPU value.
            self.cuda_sync_point = "before_finish"

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_global_tokens_per_expert = _gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert
            ).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ]
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0).to(
                torch.device("cpu"), non_blocking=True
            )
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert.to(
                torch.device("cpu"), non_blocking=True
            )

        if self.num_local_experts > 1:
            # No further synchronization is needed because torch.repeat_interleave() calls stream
            # synchronization internally when the `output_size` parameter is not provided.
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        return num_tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to local experts using AlltoAll communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
            indices (torch.Tensor): Indices of tokens assigned to experts.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(indices)

        n_bucket = 2
        rotations_shape = (
            1,
            hidden_states.size(-1),
            8,
            n_bucket // 2)
        random_rotations = torch.randn(rotations_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', hidden_states.unsqueeze(0), random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
        buckets = torch.argmax(rotated_vecs, dim=-1)
        #print(buckets.shape)
        buckets = buckets.squeeze(0).permute(1,0)
        buckets = tensor_parallel.mappings._gather_along_first_dim(buckets)        
        string = f"buckets_{torch.distributed.get_rank()}.pt"
        #torch.save(buckets, string)
        # Perform tensor parallel AlltoAll communication
        # hidden_states: [S*B/TP, H] -> [S*B, H/TP]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)

        # Permutation 1: input to AlltoAll input
        self.hiddden_shape_before_permute = hidden_states.shape
        if self.cuda_sync_point == "before_permutation_1":
            torch.cuda.current_stream().synchronize()
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            indices,
            num_out_tokens=self.num_out_tokens,
            padded_mode=self.drop_and_pad,
        )
        dispatch_buckets, _ = permute(
            buckets,
            indices,
            num_out_tokens=self.num_out_tokens,
            padded_mode=self.drop_and_pad,
        )
        string = f"dispatch_buckets_{torch.distributed.get_rank()}.pt"
        torch.save(dispatch_buckets, string)
        string = f"indices_{torch.distributed.get_rank()}.pt"
        torch.save(indices, string)
        #print(buckets.shape)
        #print(dispatch_buckets.shape)
        #assert 1==-1
        ou, permutated_local_input_tokens, self.input_split, self.expert_indices, self.t_list, self.diffs = compute_center_fn(
            permutated_local_input_tokens, dispatch_buckets, self.config.expert_model_parallel_size*self.num_local_experts, hidden_states.size(-1), self.capacity, 8)
        self.output_split = _gather_along_first_dim_expert_parallel(self.input_split).reshape(self.ep_size, self.num_experts)[
                :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].flatten()
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_parallel.all_to_all_sp2hp(hidden_states)
        
        self.ep_size = self.config.expert_model_parallel_size
        self.input_splits = self.input_split.reshape(self.ep_size, self.num_local_experts).sum(axis=1).tolist()
        self.output_splits = self.output_split.reshape(self.ep_size, self.num_local_experts).sum(axis=1).tolist()
        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        global_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
        )

        # Permutation 2: Sort alltoall output by local experts when num_local_experts > 1.
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                global_input_tokens, self.reversed_global_input_permutation_mapping = permute(
                    global_input_tokens, self.global_input_tokens_local_experts_indices
                )
            else:
                global_input_tokens = global_input_tokens.reshape(
                    self.ep_size, self.num_local_experts, self.capacity, -1
                )
                global_input_tokens = (
                    global_input_tokens.transpose(0, 1)
                    .reshape(self.num_local_experts * self.ep_size * self.capacity, -1)
                    .contiguous()
                )

        # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
        # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            global_input_tokens = tensor_parallel.all_gather_last_dim_from_tensor_parallel_region(
                global_input_tokens
            )
        if self.cuda_sync_point == "before_finish":
            torch.cuda.current_stream().synchronize()
        
        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """
        assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

        # Perform tensor parallel Reduce-Scatter
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        #h_dtype = hidden_states.dtype
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            try:
                hidden_states = tensor_parallel.reduce_scatter_last_dim_to_tensor_parallel_region(
                    hidden_states
                )
            except:
                print(torch.distributed.get_rank())
                assert 1==-1
            #hidden_states.to(h_dtype)

        # Unpermutation 2: expert output to AlltoAll input
        if self.num_local_experts > 1:
            if not self.drop_and_pad:
                hidden_states = unpermute(
                    hidden_states,
                    self.reversed_global_input_permutation_mapping,
                )
            else:
                hidden_states = hidden_states.reshape(
                    self.num_local_experts, self.ep_size, self.capacity, -1
                )
                hidden_states = (
                    hidden_states.transpose(0, 1)
                    .reshape(self.ep_size * self.num_local_experts * self.capacity, -1)
                    .contiguous()
                )

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = tensor_parallel.all_to_all(
            parallel_state.get_expert_model_parallel_group(),
            hidden_states,
            self.input_splits,
            self.output_splits,
        )
        string = f"input_tokens_{torch.distributed.get_rank()}.pt"
        torch.save(permutated_local_input_tokens, string)

        permutated_local_input_tokens = recover_token_fn(permutated_local_input_tokens, self.input_split, self.expert_indices, self.t_list, self.capacity)

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=self.probs,
            padded_mode=self.drop_and_pad,
            restore_shape=self.hiddden_shape_before_permute,
        )

        # Perform tensor parallel AlltoAll communication
        # output: [S*B, H/TP] -> [S*B/TP, H]
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            output = tensor_parallel.all_to_all_hp2sp(output)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output, None

