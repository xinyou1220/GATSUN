import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3git.sam3.model_builder import _create_vision_backbone
import sam3.perflib.fused as fused_mod

try:
    from torch_geometric.nn import GATv2Conv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARN] torch_geometric 未安裝，將使用密集 GNN 作為後備。"
          "  pip install torch_geometric")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class DeformConv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.has_deform = False
        try:
            from torchvision.ops import DeformConv2d
            self.offset_conv = nn.Conv2d(in_ch, 2 * 3 * 3, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            self.deform_conv = DeformConv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.has_deform  = True
        except ImportError:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.has_deform:
            offsets = self.offset_conv(x)
            x = self.deform_conv(x, offsets)
        else:
            x = self.conv(x)
        return self.relu(self.bn(x))


class DeformableDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        cat_ch     = in_ch + skip_ch
        self.conv1 = nn.Sequential(
            nn.Conv2d(cat_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.conv2 = DeformConv2dBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv2(self.conv1(torch.cat([skip, x], dim=1)))

class SemanticVesselPrompt(nn.Module):

    def __init__(self, n_vessels=3, feat_ch=64, n_prompt_tokens=8, n_heads=4):
        super().__init__()
        self.prompt_tokens = nn.Parameter(
            torch.randn(n_vessels, n_prompt_tokens, feat_ch) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_ch, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(feat_ch)
        self.norm2 = nn.LayerNorm(feat_ch)
        self.ffn = nn.Sequential(
            nn.Linear(feat_ch, feat_ch * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(feat_ch * 4, feat_ch), nn.Dropout(0.1))

        self.film_scale = nn.Embedding(n_vessels, feat_ch)
        self.film_shift = nn.Embedding(n_vessels, feat_ch)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_shift.weight)

    def forward(self, feat, vessel_ids):
        B, C, H, W = feat.shape
        spatial = feat.flatten(2).permute(0, 2, 1)
        prompts = self.prompt_tokens[vessel_ids]
        attn_out, _ = self.cross_attn(
            query=self.norm1(spatial), key=prompts, value=prompts)
        spatial = spatial + attn_out
        spatial = spatial + self.ffn(self.norm2(spatial))
        feat_attn = spatial.permute(0, 2, 1).reshape(B, C, H, W)

        s = self.film_scale(vessel_ids).unsqueeze(-1).unsqueeze(-1)
        b = self.film_shift(vessel_ids).unsqueeze(-1).unsqueeze(-1)
        return feat_attn * (1.0 + torch.tanh(s)) + torch.tanh(b)


class VesselTypeConditioning(nn.Module):
    def __init__(self, n_vessels=3, ch=64):
        super().__init__()
        self.embed = nn.Embedding(n_vessels, ch)
        self.scale = nn.Sequential(nn.Linear(ch, ch), nn.Tanh())
        self.shift = nn.Sequential(nn.Linear(ch, ch), nn.Tanh())

    def forward(self, feat, vessel_ids):
        emb = self.embed(vessel_ids)
        s   = self.scale(emb).unsqueeze(-1).unsqueeze(-1)
        b   = self.shift(emb).unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + s) + b


class ReIDHead(nn.Module):
    def __init__(self, in_ch=64, embed_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, embed_dim, 1, bias=False))

    def forward(self, feat, mask=None):
        pixel_embed = self.proj(feat)
        vessel_embed = None
        if mask is not None:
            mask_f = mask.float().unsqueeze(1)
            if mask_f.shape[-2:] != pixel_embed.shape[-2:]:
                mask_f = F.interpolate(mask_f, size=pixel_embed.shape[-2:],
                                       mode="nearest")
            masked = pixel_embed * mask_f
            area = mask_f.sum(dim=[2, 3]).clamp(min=1.0)
            vessel_embed = F.normalize(masked.sum(dim=[2, 3]) / area, p=2, dim=1)
        return pixel_embed, vessel_embed


class SparseGATRefinement(nn.Module):

    def __init__(self, feat_ch=64, n_heads=4, gat_layers=2,
                 k_neighbors=16, max_nodes=4096, node_threshold=0.3):
        super().__init__()
        assert HAS_PYG, (
            "SparseGATRefinement 需要 torch_geometric。\n"
            "  pip install torch_geometric\n"
            "  或設定 --use_dense_gnn 退回密集 GNN。")

        self.k              = k_neighbors
        self.max_nodes      = max_nodes
        self.node_threshold = node_threshold

        self.input_proj = nn.Linear(feat_ch, feat_ch)

        self.gat_layers = nn.ModuleList()
        self.gat_norms  = nn.ModuleList()
        for _ in range(gat_layers):
            self.gat_layers.append(GATv2Conv(
                in_channels=feat_ch,
                out_channels=feat_ch // n_heads,
                heads=n_heads,
                concat=True,
                dropout=0.1,
                add_self_loops=True,
                share_weights=False,
            ))
            self.gat_norms.append(nn.LayerNorm(feat_ch))

        self.output_proj = nn.Sequential(
            nn.Linear(feat_ch, feat_ch), nn.ReLU(inplace=True))

        self.gate = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _build_knn_graph(coords, k):

        dist = torch.cdist(coords, coords)
        _, indices = dist.topk(k + 1, dim=1, largest=False)
        indices = indices[:, 1:]
        N = coords.shape[0]
        src = torch.arange(N, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = indices.reshape(-1)
        return torch.stack([src, dst], dim=0)

    def forward(self, feat, vessel_prob):
        B, C, H, W = feat.shape
        out = feat.clone()
        gate_val = torch.sigmoid(self.gate)

        for b in range(B):
            prob_b = vessel_prob[b, 0]

            cand = prob_b > self.node_threshold
            n_cand = cand.sum().item()
            if n_cand < self.k + 1:
                continue

            ys, xs = torch.where(cand)

            if n_cand > self.max_nodes:
                _, topk_idx = prob_b[ys, xs].topk(self.max_nodes)
                ys, xs = ys[topk_idx], xs[topk_idx]

            N = ys.shape[0]
            k_actual = min(self.k, N - 1)

            node_feats = self.input_proj(feat[b, :, ys, xs].t())

            coords = torch.stack([ys.float() / H, xs.float() / W], dim=1)
            edge_index = self._build_knn_graph(coords, k_actual)

            x = node_feats
            for gat, norm in zip(self.gat_layers, self.gat_norms):
                x = x + gat(norm(x), edge_index)
                x = F.elu(x)

            x = self.output_proj(x)

            out[b, :, ys, xs] = feat[b, :, ys, xs] + gate_val * x.t()

        return out


class GraphConvLayer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.msg_convs = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=d, dilation=d, groups=ch, bias=False)
            for d in [1, 2, 4]])
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, len(self.msg_convs), 3, padding=1), nn.Sigmoid())
        self.update = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

    def forward(self, x, vessel_prob):
        gates = self.edge_gate(vessel_prob)
        msg = sum(g.unsqueeze(1) * conv(x)
                  for g, conv in zip(gates.unbind(1), self.msg_convs)
                  ) / len(self.msg_convs)
        return x + self.update(torch.cat([x, msg], dim=1))


class DenseGNNRefinement(nn.Module):
    def __init__(self, ch, n_iter=1):
        super().__init__()
        self.layers = nn.ModuleList([GraphConvLayer(ch) for _ in range(n_iter)])

    def forward(self, feat, vessel_prob):
        x = feat
        for layer in self.layers:
            x = layer(x, vessel_prob)
        return x


def _check_sam3():
    try:
        import sam3
    except ImportError:
        raise ImportError("SAM3 package not found.")


class SAM3Encoder(nn.Module):
    FPN_CH = 256; N_LEVELS = 4; PATCH_SIZE = 14; NATIVE_SIZE = 1008

    def __init__(self, checkpoint=None, freeze=True):
        super().__init__()
        _check_sam3()
        self.vision_encoder = _create_vision_backbone(
            compile_mode=None, enable_inst_interactivity=False)

        self._patch_fused_kernel()
        if checkpoint:
            self._load_checkpoint(checkpoint)
        self._frozen = freeze
        if freeze:
            self.vision_encoder.eval()
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _patch_fused_kernel():

        def _safe(act_type, linear, x):
            h = linear(x)
            if act_type is nn.GELU:   return F.gelu(h)
            elif act_type is nn.SiLU: return F.silu(h)
            elif act_type is nn.ReLU: return F.relu(h)
            return h

        import sys
        patched, failed = [], []
        for mod_name in list(sys.modules.keys()):
            if not mod_name.startswith("sam3"):
                continue
            mod = sys.modules.get(mod_name)
            if mod is None or not hasattr(mod, "addmm_act"):
                continue
            cur = getattr(mod, "addmm_act", None)
            if cur is _safe:
                continue
            try:
                setattr(mod, "addmm_act", _safe)
                patched.append(mod_name)
            except Exception as e:
                failed.append((mod_name, repr(e)))
        if patched:
            print(f"[SAM3] addmm_act patched in {len(patched)} module(s): {patched}")
        if failed:
            print(f"[SAM3] addmm_act patch FAILED in: {failed}")

    def _load_checkpoint(self, path):
        print(f"[SAM3] Loading checkpoint: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]; break
        ve_keys    = set(self.vision_encoder.state_dict().keys())
        sample_key = next(iter(ve_keys))
        discovered = ""
        for ck in state.keys():
            if ck.endswith(sample_key):
                discovered = ck[: -len(sample_key)]; break
        if discovered:
            ve_state = {k[len(discovered):]: v for k, v in state.items()
                        if k.startswith(discovered)}
        else:
            ve_state = state
            for pfx in ["backbone.vision_encoder.", "vision_encoder.", "backbone.", ""]:
                cand = {k[len(pfx):]: v for k, v in state.items()
                        if k.startswith(pfx)} if pfx else state
                if len(set(cand.keys()) & ve_keys) > len(ve_keys) * 0.3:
                    ve_state = cand; break
        m, u = self.vision_encoder.load_state_dict(ve_state, strict=False)
        print(f"[SAM3] Loaded {len(ve_state)-len(u)}, missing {len(m)}, unexpected {len(u)}")

    @staticmethod
    def _extract_fpn(raw_out):
        if not isinstance(raw_out, (list, tuple)):
            raise TypeError(f"Unexpected backbone output: {type(raw_out)}")
        for item in raw_out:
            if item is None: continue
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                tensors = []
                for sub in item:
                    if isinstance(sub, torch.Tensor):    tensors.append(sub)
                    elif hasattr(sub, "tensors"):        tensors.append(sub.tensors)
                    else:                                break
                if len(tensors) >= 4:
                    tensors.sort(key=lambda t: t.shape[-1], reverse=True)
                    return tensors[:4]
        all_t = []
        def _collect(o):
            if o is None: return
            if isinstance(o, torch.Tensor):    all_t.append(o)
            elif hasattr(o, "tensors"):        all_t.append(o.tensors)
            elif isinstance(o, (list, tuple)):
                for s in o: _collect(s)
        _collect(raw_out)
        seen, unique = set(), []
        for t in all_t:
            if t.shape not in seen:
                seen.add(t.shape); unique.append(t)
        if len(unique) >= 4:
            unique.sort(key=lambda t: t.shape[-1], reverse=True)
            return unique[:4]
        raise ValueError("Cannot extract 4 FPN levels.")

    def forward(self, x):
        in_dtype = x.dtype
        if x.shape[1] == 1: x = x.expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=(self.NATIVE_SIZE, self.NATIVE_SIZE),
                          mode="bilinear", align_corners=True)

        params_frozen = not any(p.requires_grad
                                for p in self.vision_encoder.parameters())
        use_no_grad = self._frozen or params_frozen
        if use_no_grad:
            with torch.no_grad():
                feats = self._extract_fpn(self.vision_encoder(x))
            feats = [f.detach() for f in feats]
        else:
            feats = self._extract_fpn(self.vision_encoder(x))
        feats = [f.to(in_dtype) for f in feats]
        if not hasattr(self, "_logged"):
            self._logged = True
            print(f"[SAM3] FPN: {[f.shape for f in feats]}  "
                  f"(_frozen={self._frozen}, params_frozen={params_frozen}, "
                  f"no_grad={use_no_grad})")
        return feats

    def train(self, mode=True):
        super().train(mode)
        if self._frozen: self.vision_encoder.eval()
        return self


class UNetV2(nn.Module):

    def __init__(
        self,
        checkpoint=None, 
        freeze=True,
        n_classes=2, 
        dec_channels=(256, 128, 64),
        n_vessels=3,
        use_semantic_prompt=True,
        use_sparse_gat=True,
        use_reid=True,
        n_prompt_tokens=8, 
        n_prompt_heads=4,
        gat_layers=2, 
        gat_heads=4,
        k_neighbors=16, 
        max_nodes=4096, 
        node_threshold=0.3,
        gnn_iters=3,
        reid_embed_dim=128,
        **kw,
    ):
        super().__init__()
        self.n_classes       = n_classes
        self.use_semantic_prompt = use_semantic_prompt
        self.use_sparse_gat  = use_sparse_gat and HAS_PYG
        self.use_reid        = use_reid

        fpn_ch  = SAM3Encoder.FPN_CH
        last_ch = dec_channels[-1]

        self.encoder    = SAM3Encoder(checkpoint, freeze)
        self.dec2       = DecoderBlock(fpn_ch, fpn_ch, dec_channels[0])
        self.dec1       = DecoderBlock(dec_channels[0], fpn_ch, dec_channels[1])
        self.dec0       = DeformableDecoderBlock(dec_channels[1], fpn_ch, dec_channels[2])
        self.final_conv = DoubleConv(last_ch, last_ch)

        if use_semantic_prompt:
            self.vessel_cond = SemanticVesselPrompt(
                n_vessels, last_ch, n_prompt_tokens, n_prompt_heads)
        else:
            self.vessel_cond = VesselTypeConditioning(n_vessels, last_ch)

        self.pre_seg = nn.Conv2d(last_ch, n_classes, 1)

        if self.use_sparse_gat:
            self.gnn = SparseGATRefinement(
                last_ch, gat_heads, gat_layers,
                k_neighbors, max_nodes, node_threshold)
            print(f"[GNN] SparseGAT: {gat_layers}L, {gat_heads}H, "
                  f"k={k_neighbors}, max_N={max_nodes}, thr={node_threshold}")
        else:
            self.gnn = DenseGNNRefinement(last_ch, n_iter=gnn_iters)
            print(f"[GNN] DenseGNN fallback: {gnn_iters} iters")

        self.seg_head = nn.Conv2d(last_ch, n_classes, 1)

        if use_reid:
            self.reid_head = ReIDHead(last_ch, reid_embed_dim)

    def forward(self, x, vessel_ids=None, return_reid=False):
        _, _, H, W = x.shape
        f0, f1, f2, f3 = self.encoder(x)

        d = self.dec2(f3, f2)
        d = self.dec1(d, f1)
        d = self.dec0(d, f0)
        d = self.final_conv(d)
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=True)

        if vessel_ids is not None:
            d = self.vessel_cond(d, vessel_ids)

        pre_logits = self.pre_seg(d)
        init_prob  = torch.sigmoid(pre_logits[:, 1:2])
        d_refined  = self.gnn(d, init_prob)
        seg_logits = self.seg_head(d_refined)

        if return_reid and self.use_reid:
            with torch.no_grad():
                pred_mask = (torch.sigmoid(seg_logits[:, 1]) > 0.5).float()
            pixel_embed, vessel_embed = self.reid_head(d_refined, pred_mask)
            return seg_logits, {"pixel_embed": pixel_embed,
                                "vessel_embed": vessel_embed}
        return seg_logits

    def predict_prob(self, x, vessel_ids=None):
        seg = self.forward(x, vessel_ids)
        if isinstance(seg, tuple): seg = seg[0]
        return torch.sigmoid(seg[:, 1])

    def backbone_parameters(self):
        return self.encoder.vision_encoder.parameters()

    def decoder_parameters(self):
        exclude = {"encoder", "reid_head"}
        for name, module in self.named_modules():
            if any(name.startswith(ex) for ex in exclude):
                continue
            for p in module.parameters(recurse=False):
                yield p

    def reid_parameters(self):
        if self.use_reid:
            yield from self.reid_head.parameters()


ALPHA_BOUNDARY     = 4.0
SIGMA_BOUNDARY     = 3.0
ALPHA_SKELETON     = 4.0
SKEL_DILATE_KERNEL = 5
GT_SKEL_ITERS      = 20


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs  = torch.sigmoid(logits.float()).contiguous().view(logits.size(0), -1)
        target = target.float().contiguous().view(target.size(0), -1)
        inter  = (probs * target).sum(dim=1)
        dice   = (2.0 * inter + self.smooth) / (probs.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class SoftSkeletonize(nn.Module):
    def __init__(self, iter_num: int = 20):
        super().__init__()
        self.iter_num = iter_num

    @staticmethod
    def soft_erode(img):
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.minimum(p1, p2)

    @staticmethod
    def soft_dilate(img):
        return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

    @classmethod
    def soft_open(cls, img):
        return cls.soft_dilate(cls.soft_erode(img))

    def forward(self, img):
        img  = img.clamp(0.0, 1.0)
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.iter_num):
            img   = self.soft_erode(img)
            img1  = self.soft_open(img)
            delta = F.relu(img - img1)
            skel  = skel + F.relu(delta - skel * delta)
        return skel.clamp(0.0, 1.0)


class ClDiceLoss(nn.Module):

    def __init__(self, iter_num: int = 10, smooth: float = 1e-6):
        super().__init__()
        self.soft_skel = SoftSkeletonize(iter_num)
        self.smooth    = smooth

    def forward(self, logits, target):
        # AMP-safe：上 fp32
        logits = logits.float()
        target = target.float().to(device=logits.device)
        probs  = torch.sigmoid(logits)
        sp = self.soft_skel(probs)
        st = self.soft_skel(target)
        tprec = ((sp * target).sum(dim=(1, 2, 3)) + self.smooth) \
              / (sp.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = ((st * probs).sum(dim=(1, 2, 3)) + self.smooth) \
              / (st.sum(dim=(1, 2, 3)) + self.smooth)
        cl = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl.mean()

@torch.no_grad()
def _boundary_distance(target: torch.Tensor) -> torch.Tensor:

    target_f = target.float()
    eroded   = -F.max_pool2d(-target_f, kernel_size=3, stride=1, padding=1)
    boundary = (target_f - eroded).clamp(0.0, 1.0)
    dilated  = F.max_pool2d(target_f, kernel_size=3, stride=1, padding=1)
    outer    = (dilated - target_f).clamp(0.0, 1.0)
    bnd      = ((boundary + outer) > 0).float()

    max_steps = max(1, int(SIGMA_BOUNDARY * 4))
    dist = torch.full_like(target_f, float(max_steps))
    cur  = bnd
    dist = torch.where(cur > 0, torch.zeros_like(dist), dist)
    for k in range(1, max_steps + 1):
        cur  = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
        dist = torch.where((cur > 0) & (dist >= max_steps),
                           torch.full_like(dist, float(k)), dist)
    return dist


@torch.no_grad()
def _gt_skeleton_band(target: torch.Tensor) -> torch.Tensor:

    target_f = target.float()
    skel = SoftSkeletonize(iter_num=GT_SKEL_ITERS)(target_f)
    skel = (skel > 0.5).float()
    pad  = SKEL_DILATE_KERNEL // 2
    return F.max_pool2d(skel, kernel_size=SKEL_DILATE_KERNEL,
                         stride=1, padding=pad)


@torch.no_grad()
def build_weight_map(target: torch.Tensor,
                     skeleton: torch.Tensor = None) -> torch.Tensor:

    target_f = target.float()
    d   = _boundary_distance(target_f)
    w_b = 1.0 + ALPHA_BOUNDARY * torch.exp(-d / SIGMA_BOUNDARY)
    if skeleton is None:
        skeleton = _gt_skeleton_band(target_f)
    w_s = ALPHA_SKELETON * skeleton.float().to(target.device)
    return w_b + w_s


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                              weight: torch.Tensor) -> torch.Tensor:

    bce_pp = F.binary_cross_entropy_with_logits(logits, target.float(),
                                                  reduction="none")
    w      = weight.float()
    return (bce_pp * w).sum() / w.sum().clamp_min(1.0)


def weighted_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                       weight: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:

    probs  = torch.sigmoid(logits.float())
    target = target.float()
    weight = weight.float()
    B = logits.shape[0]
    p = probs.view(B, -1)
    t = target.view(B, -1)
    w = weight.view(B, -1)
    inter = (w * p * t).sum(dim=1)
    denom = (w * p).sum(dim=1) + (w * t).sum(dim=1)
    dice  = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, vessel_ids):
        if embeddings is None or embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device
                                if embeddings is not None else "cpu")
        B = embeddings.shape[0]
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature
        labels = vessel_ids.unsqueeze(0) == vessel_ids.unsqueeze(1)
        labels.fill_diagonal_(False)
        if not labels.any():
            return torch.tensor(0.0, device=embeddings.device)
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        sim    = sim[mask].reshape(B, B - 1)
        labels = labels[mask].reshape(B, B - 1)
        log_prob = F.log_softmax(sim, dim=1)
        n_pos = labels.float().sum(dim=1).clamp(min=1)
        return -(log_prob * labels.float()).sum(dim=1).div(n_pos).mean()


class SegLossV2(nn.Module):

    def __init__(self,
                 lambda_cldice:    float = 0.5,
                 lambda_reid:      float = 0.1,
                 cldice_iter:      int   = 10,
                 reid_temperature: float = 0.07):
        super().__init__()
        self.lambda_cldice = lambda_cldice
        self.lambda_reid   = lambda_reid
        self.dice          = DiceLoss()
        self.cldice        = ClDiceLoss(iter_num=cldice_iter)
        if lambda_reid > 0:
            self.infonce = InfoNCELoss(temperature=reid_temperature)

    @staticmethod
    def _prep(seg_logits: torch.Tensor,
              mask_target: torch.Tensor):

        fg_logits = seg_logits[:, 1:2]
        if mask_target.dim() == 3:
            target = mask_target.unsqueeze(1)
        elif mask_target.dim() == 4:
            target = mask_target
        else:
            raise ValueError(f"mask_target dim must be 3 or 4, got {mask_target.dim()}")
        target = target.to(dtype=fg_logits.dtype)
        return fg_logits, target

    def forward(self, seg_logits, mask_target,
                reid_dict=None, vessel_ids=None,
                skeleton=None):

        fg_logits, target = self._prep(seg_logits, mask_target)

        weight = build_weight_map(target, skeleton=skeleton)

        l_bce    = weighted_bce_with_logits(fg_logits, target, weight)
        l_dice_w = weighted_dice_loss(fg_logits, target, weight)
        l_cldice = self.cldice(fg_logits, target)

        total = l_bce + l_dice_w + self.lambda_cldice * l_cldice

        result = {
            "total":       total,
            "bce_loss":    l_bce.detach(),
            "dice_loss":   l_dice_w.detach(),
            "cldice_loss": l_cldice.detach(),
        }

        if (self.lambda_reid > 0 and reid_dict is not None
                and reid_dict.get("vessel_embed") is not None
                and vessel_ids is not None):
            loss_reid = self.infonce(reid_dict["vessel_embed"], vessel_ids)
            total = total + self.lambda_reid * loss_reid
            result["reid_loss"] = loss_reid.detach()
            result["total"]     = total

        return result


def soft_erode(img, k=3):
    return -F.max_pool2d(-img, kernel_size=k, stride=1, padding=k // 2)

def soft_dilate(img, k=3):
    return F.max_pool2d(img, kernel_size=k, stride=1, padding=k // 2)

def soft_open(img, k=3):
    return soft_dilate(soft_erode(img, k), k)

def soft_skeletonize(img, n_iter=15, k=3):
    cur  = img.clone()
    skel = torch.zeros_like(img)
    for _ in range(n_iter):
        skel = torch.max(skel, F.relu(cur - soft_open(cur, k)))
        cur  = soft_erode(cur, k)
    return skel

UNet = UNetV2
SegLoss = SegLossV2