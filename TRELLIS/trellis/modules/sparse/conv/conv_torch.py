import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import SparseTensor


def _triple(value):
    return tuple(value) if isinstance(value, (list, tuple)) else (value, value, value)


def _same_padding(kernel_size, dilation):
    kernel_size = _triple(kernel_size)
    dilation = _triple(dilation)
    return tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation))


def _dense_from_sparse(x: SparseTensor) -> torch.Tensor:
    return x.dense()


def _gather_submanifold(dense: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    idx = coords.long()
    return dense[idx[:, 0], :, idx[:, 1], idx[:, 2], idx[:, 3]]


def _gather_single_batch(dense: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    idx = coords.long()
    return dense[0, :, idx[:, 0], idx[:, 1], idx[:, 2]].transpose(0, 1)


def _active_coords_from_mask(mask: torch.Tensor) -> torch.Tensor:
    active = torch.nonzero(mask[:, 0] > 0, as_tuple=False)
    if active.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.int32, device=mask.device)
    return active.to(torch.int32)


def _crop_bounds(coords: torch.Tensor, spatial_shape, kernel_size, dilation, padding):
    coords = coords.long()
    spatial_shape = torch.tensor(spatial_shape, dtype=torch.long, device=coords.device)
    padding = torch.tensor(padding, dtype=torch.long, device=coords.device)
    kernel_size = torch.tensor(kernel_size, dtype=torch.long, device=coords.device)
    dilation = torch.tensor(dilation, dtype=torch.long, device=coords.device)
    right_halo = (kernel_size - 1) * dilation - padding

    crop_min = torch.clamp(coords.min(dim=0).values - padding, min=0)
    crop_max = torch.minimum(coords.max(dim=0).values + right_halo, spatial_shape - 1)
    return crop_min, crop_max


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.padding = _same_padding(kernel_size, dilation) if padding is None else _triple(padding)
        self.subm = all(s == 1 for s in self.stride) and padding is None
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias,
        )
        self.out_channels = out_channels

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "conv.weight"
        weight = state_dict.get(weight_key)
        if weight is not None and weight.shape != self.conv.weight.shape:
            if (
                weight.ndim == 5
                and weight.shape[0] == self.conv.out_channels
                and weight.shape[-1] == self.conv.in_channels
            ):
                state_dict[weight_key] = weight.permute(0, 4, 1, 2, 3).contiguous()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _forward_subm_cropped(self, x: SparseTensor) -> torch.Tensor:
        if x.coords.numel() == 0:
            return x.feats.new_zeros((0, self.out_channels))

        output = x.feats.new_empty((x.feats.shape[0], self.out_channels))
        spatial_shape = x.data.spatial_shape

        for batch_idx in range(x.shape[0]):
            layout = x.layout[batch_idx]
            if layout.stop == layout.start:
                continue

            coords = x.coords[layout, 1:]
            feats = x.feats[layout]
            crop_min, crop_max = _crop_bounds(coords, spatial_shape, self.kernel_size, self.dilation, self.padding)
            crop_shape = (crop_max - crop_min + 1).tolist()
            local_coords = (coords.long() - crop_min).long()

            dense = feats.new_zeros((1, feats.shape[1], *crop_shape))
            dense[0, :, local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = feats.transpose(0, 1)
            out_dense = self.conv(dense)
            output[layout] = _gather_single_batch(out_dense, local_coords)

        return output

    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.subm:
            new_coords = x.coords
            new_feats = self._forward_subm_cropped(x)
            new_layout = x.layout
        else:
            dense = _dense_from_sparse(x)
            out_dense = self.conv(dense)
            occupancy = torch.zeros(
                (dense.shape[0], 1, *dense.shape[2:]),
                dtype=dense.dtype,
                device=dense.device,
            )
            if x.coords.numel():
                coords = x.coords.long()
                occupancy[coords[:, 0], 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1
            weight = torch.ones((1, 1, *self.kernel_size), dtype=dense.dtype, device=dense.device)
            active_mask = F.conv3d(
                occupancy,
                weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            new_coords = _active_coords_from_mask(active_mask)
            new_feats = _gather_submanifold(out_dense, new_coords) if new_coords.numel() else out_dense.new_zeros((0, self.out_channels))
            new_layout = None

        out = SparseTensor(
            new_feats,
            new_coords,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.dilation = _triple(dilation)
        self.padding = _same_padding(kernel_size, dilation)
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias,
        )
        self.out_channels = out_channels

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + "conv.weight"
        weight = state_dict.get(weight_key)
        if weight is not None and weight.shape != self.conv.weight.shape:
            if (
                weight.ndim == 5
                and weight.shape[0] == self.conv.out_channels
                and weight.shape[-1] == self.conv.in_channels
            ):
                state_dict[weight_key] = weight.permute(4, 0, 1, 2, 3).contiguous()
            elif (
                weight.ndim == 5
                and weight.shape[0] == self.conv.in_channels
                and weight.shape[-1] == self.conv.out_channels
            ):
                state_dict[weight_key] = weight.permute(0, 4, 1, 2, 3).contiguous()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: SparseTensor) -> SparseTensor:
        dense = _dense_from_sparse(x)
        out_dense = self.conv(dense)

        occupancy = torch.zeros(
            (dense.shape[0], 1, *dense.shape[2:]),
            dtype=dense.dtype,
            device=dense.device,
        )
        if x.coords.numel():
            coords = x.coords.long()
            occupancy[coords[:, 0], 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1
        weight = torch.ones((1, 1, *self.kernel_size), dtype=dense.dtype, device=dense.device)
        active_mask = F.conv_transpose3d(
            occupancy,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        new_coords = _active_coords_from_mask(active_mask)
        new_feats = _gather_submanifold(out_dense, new_coords) if new_coords.numel() else out_dense.new_zeros((0, self.out_channels))

        out = SparseTensor(
            new_feats,
            new_coords,
            shape=torch.Size([x.shape[0], self.out_channels]),
            layout=None,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out
