"""ResNet-18 encoder, gated-attention MIL pooling, and the single- and multi-sequence models.

Submodule names (encoder, shared_encoder, mil, shared_mil, seq_attention, joint_attention,
clinical_encoder, classifier, ...) define the checkpoint keys and are kept fixed.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

STEP_SINGLE, STEP_MIL, STEP_FUSION, STEP_COMBINED = 1, 2, 3, 4
FEATURE_DIM = 512


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Channel + spatial attention on the last feature map (Woo et al., 2018). Off by default:
    in our ablation it increased overfitting at this sample size."""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        spatial_weights = self.sa(x)
        return x * spatial_weights, spatial_weights


class ClinicalEncoder(nn.Module):
    """Small embedding for [age, sex, laterality] before concatenation with image features."""

    def __init__(self, in_dim=3, out_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        return self.net(x)


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al., 2018) with a learned positional embedding.
    Pools over slices (index = slice position) or over sequences (index = sequence order)."""

    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=128, max_slices=11):
        super().__init__()
        self.L = feature_dim
        self.D = hidden_dim
        self.pos_emb = nn.Embedding(max_slices, self.L)
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_w = nn.Linear(self.D, 1)

    def forward(self, x, slice_indices=None):
        """x: [B, N, L] -> pooled [B, L], weights [B, N, 1]."""
        x_att = x + self.pos_emb(slice_indices) if slice_indices is not None else x
        A = self.attention_w(self.attention_V(x_att) * self.attention_U(x_att))
        A = torch.softmax(A, dim=1)
        return torch.sum(x * A, dim=1), A


class JointSliceSequenceAttention(nn.Module):
    """One gated-attention softmax over the whole (slice x sequence) grid, so the slice
    weighting may depend on the sequence and vice versa. Positions enter additively
    (slice embedding + sequence embedding), which needs S + M rows instead of S * M."""

    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=64, max_slices=5, max_sequences=3):
        super().__init__()
        self.L, self.D = feature_dim, hidden_dim
        self.slice_emb = nn.Embedding(max_slices, feature_dim)
        self.seq_emb = nn.Embedding(max_sequences, feature_dim)
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_w = nn.Linear(self.D, 1)

    def forward(self, x, slice_indices=None):
        """x: [B, S, M, L] -> pooled [B, L], weights [B, S, M]."""
        B, S, M, L = x.shape
        if slice_indices is None:
            slice_indices = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        seq_idx = torch.arange(M, device=x.device)
        pe = self.slice_emb(slice_indices).unsqueeze(2) + self.seq_emb(seq_idx).view(1, 1, M, L)
        h = x + pe
        A = self.attention_w(self.attention_V(h) * self.attention_U(h))
        A = torch.softmax(A.view(B, S * M, 1), dim=1).view(B, S, M, 1)
        return (x * A).sum(dim=(1, 2)), A.squeeze(-1)


def set_bn_eval(module):
    if "BatchNorm" in module.__class__.__name__:
        module.eval()


class ResNetEncoder(nn.Module):
    """ImageNet ResNet-18 without its head. freeze_depth: 'layer4' trains only the last
    block, 'layer3' the last two, 'none' everything."""

    UNFROZEN = {"none": ["features"], "layer4": ["features.7"], "layer3": ["features.6", "features.7"]}

    def __init__(self, freeze_depth="layer4", use_cbam=False):
        super().__init__()
        self.use_cbam = use_cbam
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        if use_cbam:
            self.cbam = CBAM(in_planes=FEATURE_DIM)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        for p in self.parameters():
            p.requires_grad = False
        unfreeze = self.UNFROZEN[freeze_depth] + (["cbam"] if use_cbam else [])
        for name, p in self.named_parameters():
            if any(layer in name for layer in unfreeze):
                p.requires_grad = True

    def forward(self, x):
        f_map = self.features(x)
        spatial_weights = None
        if self.use_cbam:
            f_map, spatial_weights = self.cbam(f_map)
        return self.global_pool(f_map).flatten(1), spatial_weights


def _head(in_features, num_classes):
    return nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))


def _fuse(feats, fusion_type, seq_attention=None):
    """feats: list of [N, 512] per sequence -> fused [N, 512 or 512*M], weights or None."""
    if fusion_type == "concat":
        return torch.cat(feats, dim=1), None
    if fusion_type == "add":
        return torch.stack(feats).sum(dim=0), None
    if fusion_type == "seq_attention":
        stack = torch.stack(feats, dim=1)  # [N, M, 512]
        n, m = stack.shape[:2]
        idx = torch.arange(m, device=stack.device).unsqueeze(0).expand(n, -1)
        return seq_attention(stack, idx)
    raise ValueError(f"Unknown fusion_type '{fusion_type}'")


class _ClinicalMixin:
    def _add_clinical(self, feats, clinical):
        if self.use_clinical and clinical is not None:
            return torch.cat([feats, self.clinical_encoder(clinical)], dim=1)
        return feats


# ---------------------------------------------------------------------------
# Step 1: single slice, single sequence
# ---------------------------------------------------------------------------
class SingleSliceModel(_ClinicalMixin, nn.Module):
    def __init__(self, num_classes, use_clinical=False, freeze_depth="layer4", use_cbam=False, clinical_dim=16):
        super().__init__()
        self.encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        self.use_clinical = use_clinical
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(FEATURE_DIM + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images, clinical=None):
        feats, spatial_map = self.encoder(images)
        return self.classifier(self._add_clinical(feats, clinical)), spatial_map


# ---------------------------------------------------------------------------
# Step 2: multi-slice MIL, single sequence
# ---------------------------------------------------------------------------
class MILModel(_ClinicalMixin, nn.Module):
    def __init__(self, num_classes, freeze_depth="layer4", use_cbam=False, mil_type="attention",
                 use_clinical=False, mil_hidden_dim=128, max_slices=11, clinical_dim=16):
        super().__init__()
        self.encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        self.mil_type = mil_type
        self.use_clinical = use_clinical
        if mil_type == "attention":
            self.mil = GatedAttentionMIL(FEATURE_DIM, mil_hidden_dim, max_slices)
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(FEATURE_DIM + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images, clinical=None, slice_indices=None):
        B, S, C, H, W = images.shape
        feats, _ = self.encoder(images.view(B * S, C, H, W))
        feats = feats.view(B, S, -1)
        if self.mil_type == "attention":
            pooled, weights = self.mil(feats, slice_indices)
        elif self.mil_type == "max":
            pooled, weights = torch.max(feats, dim=1)[0], None
        else:
            pooled, weights = torch.mean(feats, dim=1), None
        return self.classifier(self._add_clinical(pooled, clinical)), weights


# ---------------------------------------------------------------------------
# Step 3: single slice, multi-sequence fusion
# ---------------------------------------------------------------------------
class EarlyFusionModel(_ClinicalMixin, nn.Module):
    """Three sequences become the three input channels."""

    def __init__(self, num_classes, sequences, freeze_depth="layer4", use_cbam=False, use_clinical=False,
                 clinical_dim=16):
        super().__init__()
        if len(sequences) != 3:
            raise ValueError("Early fusion needs exactly 3 sequences.")
        self.sequences = list(sequences)
        self.use_clinical = use_clinical
        self.encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(FEATURE_DIM + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images_dict, clinical=None):
        fused_input = torch.cat([images_dict[s][:, 0:1] for s in self.sequences], dim=1)
        feats, spatial_map = self.encoder(fused_input)
        return self.classifier(self._add_clinical(feats, clinical)), spatial_map


class LateFusionModel(_ClinicalMixin, nn.Module):
    """Shared encoder per sequence, then concat / add / sequence attention.
    Sequence attention lets the weighting of sequences vary per patient."""

    def __init__(self, num_classes, sequences, fusion_type="concat", freeze_depth="layer4", use_cbam=False,
                 use_clinical=False, seq_attn_hidden_dim=32, clinical_dim=16):
        super().__init__()
        self.sequences = list(sequences)
        self.fusion_type = fusion_type
        self.use_clinical = use_clinical
        self.shared_encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        if fusion_type == "seq_attention":
            self.seq_attention = GatedAttentionMIL(FEATURE_DIM, seq_attn_hidden_dim, len(self.sequences))
        fused_dim = FEATURE_DIM * len(self.sequences) if fusion_type == "concat" else FEATURE_DIM
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(fused_dim + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images_dict, clinical=None):
        feats, aux = [], {}
        for seq in self.sequences:
            f, s_map = self.shared_encoder(images_dict[seq])
            feats.append(f)
            aux[seq] = s_map
        fused, seq_weights = _fuse(feats, self.fusion_type, getattr(self, "seq_attention", None))
        if seq_weights is not None:
            aux["__seq_attention__"] = seq_weights
        return self.classifier(self._add_clinical(fused, clinical)), aux


# ---------------------------------------------------------------------------
# Step 4: multi-slice, multi-sequence
# ---------------------------------------------------------------------------
class EarlyFusionMIL(_ClinicalMixin, nn.Module):
    def __init__(self, num_classes, sequences, freeze_depth="layer4", use_cbam=False, mil_type="attention",
                 max_slices=11, use_clinical=False, mil_hidden_dim=128, clinical_dim=16):
        super().__init__()
        if len(sequences) != 3:
            raise ValueError("Early fusion needs exactly 3 sequences.")
        self.sequences = list(sequences)
        self.mil_type = mil_type
        self.use_clinical = use_clinical
        self.encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        if mil_type == "attention":
            self.mil = GatedAttentionMIL(FEATURE_DIM, mil_hidden_dim, max_slices)
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(FEATURE_DIM + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images_dict, clinical=None, slice_indices=None):
        B, S, C, H, W = images_dict[self.sequences[0]].shape
        fused = torch.cat([images_dict[s][:, :, 0:1] for s in self.sequences], dim=2)
        feats, _ = self.encoder(fused.view(B * S, 3, H, W))
        feats = feats.view(B, S, -1)
        if self.mil_type == "attention":
            pooled, weights = self.mil(feats, slice_indices)
        else:
            pooled, weights = torch.mean(feats, dim=1), None
        return self.classifier(self._add_clinical(pooled, clinical)), weights


class LateFusionMIL(_ClinicalMixin, nn.Module):
    """Slice attention per sequence (shared head), then fusion across sequences."""

    def __init__(self, num_classes, sequences, freeze_depth="layer4", use_cbam=False, fusion_type="concat",
                 mil_type="attention", max_slices=11, use_clinical=False, mil_hidden_dim=128,
                 seq_attn_hidden_dim=32, clinical_dim=16):
        super().__init__()
        self.sequences = list(sequences)
        self.fusion_type = fusion_type
        self.mil_type = mil_type
        self.use_clinical = use_clinical
        self.shared_encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        if mil_type == "attention":
            self.shared_mil = GatedAttentionMIL(FEATURE_DIM, mil_hidden_dim, max_slices)
        if fusion_type == "seq_attention":
            self.seq_attention = GatedAttentionMIL(FEATURE_DIM, seq_attn_hidden_dim, len(self.sequences))
        fused_dim = FEATURE_DIM * len(self.sequences) if fusion_type == "concat" else FEATURE_DIM
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(fused_dim + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images_dict, clinical=None, slice_indices=None):
        B, S, C, H, W = images_dict[self.sequences[0]].shape
        seq_features, aux = [], {}
        for seq in self.sequences:
            feats, _ = self.shared_encoder(images_dict[seq].view(B * S, C, H, W))
            feats = feats.view(B, S, -1)
            if self.mil_type == "attention":
                pooled, weights = self.shared_mil(feats, slice_indices)
            else:
                pooled, weights = torch.mean(feats, dim=1), None
            seq_features.append(pooled)
            aux[seq] = weights
        fused, seq_weights = _fuse(seq_features, self.fusion_type, getattr(self, "seq_attention", None))
        if seq_weights is not None:
            aux["__seq_attention__"] = seq_weights
        return self.classifier(self._add_clinical(fused, clinical)), aux


class MidFusionMIL(_ClinicalMixin, nn.Module):
    """Sequences fused per slice first, then one slice-attention pass. With sequence
    attention, the sequence weighting is computed separately at every slice."""

    def __init__(self, num_classes, sequences, freeze_depth="layer4", use_cbam=False, fusion_type="concat",
                 mil_type="attention", max_slices=11, use_clinical=False, mil_hidden_dim=128,
                 seq_attn_hidden_dim=32, clinical_dim=16):
        super().__init__()
        self.sequences = list(sequences)
        self.fusion_type = fusion_type
        self.mil_type = mil_type
        self.use_clinical = use_clinical
        self.shared_encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        if fusion_type == "concat":
            self.slice_fusion_project = nn.Sequential(
                nn.Linear(FEATURE_DIM * len(self.sequences), FEATURE_DIM), nn.ReLU(), nn.Dropout(0.3))
        elif fusion_type == "seq_attention":
            self.seq_attention = GatedAttentionMIL(FEATURE_DIM, seq_attn_hidden_dim, len(self.sequences))
        if mil_type == "attention":
            self.shared_mil = GatedAttentionMIL(FEATURE_DIM, mil_hidden_dim, max_slices)
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(FEATURE_DIM + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images_dict, clinical=None, slice_indices=None):
        per_seq = []
        for seq in self.sequences:
            x = images_dict[seq]
            B, S, C, H, W = x.shape
            feats, _ = self.shared_encoder(x.view(B * S, C, H, W))
            per_seq.append(feats.view(B, S, -1))

        aux = {}
        if self.fusion_type == "concat":
            fused_slices = self.slice_fusion_project(torch.cat(per_seq, dim=-1))
        elif self.fusion_type == "add":
            fused_slices = torch.stack(per_seq).sum(dim=0)
        elif self.fusion_type == "seq_attention":
            stacked = torch.stack(per_seq, dim=2)  # [B, S, M, 512]
            M = stacked.shape[2]
            flat = stacked.reshape(B * S, M, FEATURE_DIM)
            idx = torch.arange(M, device=flat.device).unsqueeze(0).expand(B * S, -1)
            fused_flat, seq_w = self.seq_attention(flat, idx)
            fused_slices = fused_flat.view(B, S, FEATURE_DIM)
            aux["__seq_attention__"] = seq_w.view(B, S, M).mean(dim=1).unsqueeze(-1)
            aux["__seq_attention_per_slice__"] = seq_w.view(B, S, M)
        else:
            raise ValueError(f"Unknown fusion_type '{self.fusion_type}'")

        if self.mil_type == "attention":
            pooled, weights = self.shared_mil(fused_slices, slice_indices)
        else:
            pooled, weights = torch.mean(fused_slices, dim=1), None
        aux["fused_attention"] = weights
        return self.classifier(self._add_clinical(pooled, clinical)), aux


class JointAttentionMIL(_ClinicalMixin, nn.Module):
    """Shared encoder over every (slice, sequence) image, then one joint attention pass."""

    def __init__(self, num_classes, sequences, freeze_depth="layer4", use_cbam=False, max_slices=11,
                 use_clinical=False, mil_hidden_dim=64, clinical_dim=16):
        super().__init__()
        self.sequences = list(sequences)
        self.use_clinical = use_clinical
        self.shared_encoder = ResNetEncoder(freeze_depth=freeze_depth, use_cbam=use_cbam)
        self.joint_attention = JointSliceSequenceAttention(FEATURE_DIM, mil_hidden_dim, max_slices,
                                                           len(self.sequences))
        self.clinical_encoder = ClinicalEncoder(3, clinical_dim) if use_clinical else None
        self.classifier = _head(FEATURE_DIM + (clinical_dim if use_clinical else 0), num_classes)

    def forward(self, images_dict, clinical=None, slice_indices=None):
        per_seq = []
        for seq in self.sequences:
            x = images_dict[seq]
            B, S, C, H, W = x.shape
            f, _ = self.shared_encoder(x.view(B * S, C, H, W))
            per_seq.append(f.view(B, S, -1))

        pooled, A = self.joint_attention(torch.stack(per_seq, dim=2), slice_indices)
        # Marginals, so the slice/sequence diagnostics used for the factorised models still apply.
        aux = {"__joint_attention__": A,
               "__seq_attention__": A.sum(dim=1).unsqueeze(-1),
               "fused_attention": A.sum(dim=2).unsqueeze(-1)}
        for i, seq in enumerate(self.sequences):
            aux[seq] = A[:, :, i].unsqueeze(-1)
        return self.classifier(self._add_clinical(pooled, clinical)), aux


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(step, num_classes, sequences, model_type="single", fusion_type="concat", mil_type="mean",
                use_clinical=False, num_slices=5, freeze_depth="layer4", use_cbam=False, mil_hidden_dim=64,
                seq_attn_hidden_dim=32, clinical_dim=16) -> nn.Module:
    common = dict(freeze_depth=freeze_depth, use_cbam=use_cbam, use_clinical=use_clinical,
                  clinical_dim=clinical_dim)
    if step == STEP_SINGLE:
        return SingleSliceModel(num_classes, **common)
    if step == STEP_MIL:
        return MILModel(num_classes, mil_type=mil_type, mil_hidden_dim=mil_hidden_dim, max_slices=num_slices,
                        **common)
    if step == STEP_FUSION:
        if model_type == "early":
            return EarlyFusionModel(num_classes, sequences, **common)
        return LateFusionModel(num_classes, sequences, fusion_type=fusion_type,
                               seq_attn_hidden_dim=seq_attn_hidden_dim, **common)
    if step == STEP_COMBINED:
        if model_type == "joint":
            return JointAttentionMIL(num_classes, sequences, max_slices=num_slices, mil_hidden_dim=mil_hidden_dim,
                                     **common)
        if model_type == "early":
            return EarlyFusionMIL(num_classes, sequences, mil_type=mil_type, max_slices=num_slices,
                                  mil_hidden_dim=mil_hidden_dim, **common)
        cls = MidFusionMIL if model_type == "mid" else LateFusionMIL
        return cls(num_classes, sequences, fusion_type=fusion_type, mil_type=mil_type, max_slices=num_slices,
                   mil_hidden_dim=mil_hidden_dim, seq_attn_hidden_dim=seq_attn_hidden_dim, **common)
    raise ValueError(f"Unknown step {step}")


def model_kwargs(cfg) -> dict:
    """Architecture hyper-parameters from the config, as accepted by build_model."""
    m = cfg.model
    return dict(num_slices=cfg.data.num_slices, freeze_depth=m.freeze_depth, use_cbam=m.use_cbam,
                mil_hidden_dim=m.mil_hidden_dim, seq_attn_hidden_dim=m.seq_attn_hidden_dim,
                clinical_dim=m.clinical_embed_dim)
