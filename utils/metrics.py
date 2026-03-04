import logging
from prettytable import PrettyTable

import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    """
    similarity: torch.Tensor [num_query, num_gallery]
    q_pids/g_pids: torch.Tensor [num_query] / [num_gallery]
    """
    # Use PyTorch operations (expects similarity as Tensor or array-like)
    if get_mAP:
        similarity = torch.tensor(similarity)
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # accelerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )

    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    # CMC
    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True
    )
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    row = [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]
    return (row, indices) if retur_indices else row


class Evaluator:
    """
    kNC REMOVED.
    Inference prompt (PromptSG-style):
      - simplified: fixed text query for cross-attn (fast)
      - composed: per-image pseudo-token -> prompt learner -> text encoder (slow)
    Similarity is computed between:
      query_text_embedding (from caption)  vs  gallery_embedding (after cross-attn bottleneck)
    """
    def __init__(self, img_loader, txt_loader, refer_loader, args):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query
        self.refer_loader = refer_loader  # unused (kept for backward-compat)
        self.logger = logging.getLogger("LPNC.eval")
        self.args = args

    def _build_fixed_text_feature(self, model, device):
        """
        Build a fixed prompt embedding used as Query in cross-attention
        when infer_prompt == 'simplified'.
        We avoid depending on external tokenizers here by using model.encode_text
        on a tokenized fixed string if available; otherwise fallback to composed.
        """
        # If your repo later adds args.fixed_prompt, it will be used
        fixed_prompt = getattr(self.args, "fixed_prompt", "A photo of a person")

        # Try to reuse the same tokenizer pipeline as TextDataset (SimpleTokenizer).
        # We implement a tiny tokenizer wrapper here to avoid importing datasets.*.
        try:
            from utils.simple_tokenizer import SimpleTokenizer
            tokenizer = SimpleTokenizer()
            sot = tokenizer.encoder["<|startoftext|>"]
            eot = tokenizer.encoder["<|endoftext|>"]
            tokens = [sot] + tokenizer.encode(fixed_prompt) + [eot]
            text_length = getattr(self.args, "text_length", 77)
            ids = torch.zeros(text_length, dtype=torch.long)
            if len(tokens) > text_length:
                tokens = tokens[:text_length]
                tokens[-1] = eot
            ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            ids = ids.unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model.encode_text(ids)  # (1, 512)
            return feat.float()
        except Exception:
            return None  # fallback

    def _process_images(self, model, img_loader, infer_prompt, fixed_text_feature, device):
        """
        Process images through the cross-attention pipeline.

        Composed:
          img -> encode_image1 -> image_tokens
          S*_global = img2text(CLS)
          S*_local  = local_pseudo(patch_tokens)   (no t_feats / W)
          S* = S*_global + S*_local
          "A photo of a S* person" -> text_encoder -> cross_attn(text, image) -> BN

        Simplified:
          fixed prompt "A photo of a person" -> cross_attn(text, image) -> BN
        """
        pids_list, feats_list = [], []
        for pid, img in img_loader:
            img = img.to(device)
            with torch.no_grad():
                image_tokens = model.encode_image1(img)       # (B, 1+M, D)
                img_tokens = model.encode_image(img)
                bsz = image_tokens.shape[0]

                if infer_prompt == "composed":
                    # S*_global from CLS token
                    i_feats = image_tokens[:, 0, :].float()   # (B, D)
                    # keep IM2TEXT input in float to match module parameter dtypes
                    s_global = model.img2text(i_feats)  # (B, D)

                    # S*_local from patch tokens (no t_feats refinement, no W)
                    patch_feats = image_tokens[:, 1:, :].float()  # (B, M, D)

                    # New LPNC API: run local queries cross-attention + FFN in model,
                    # then map averaged query outputs to S*_local via local_mapping.
                    avg_p = model.cross_former_local(patch_feats)  # (B, D)
                    s_local = model.local_mapping(avg_p)          # (B, D)

                    # S* = S*_global + S*_local (keep both in float dtype)
                    pseudo_token = s_global + s_local       # (B, D)

                    # Evaluation: avoid mixed-precision/autocast to keep dtypes stable
                    prompts = model.prompt_learner(pseudo_token)
                    text_feature = model.text_encoder(
                        prompts, model.prompt_learner.tokenized_prompts
                    )
                else:
                    # simplified: "A photo of a person" (no mapping network)
                    text_feature = fixed_text_feature.expand(bsz, -1)

                # cross-attn  (Q=text_feature, K/V=image_tokens)
                cross_x = model.cross_former(
                    text_feature.unsqueeze(1), image_tokens, image_tokens
                )
                cross_x_bn = model.bottleneck_proj(cross_x.squeeze(1))  # (B, D)

            pids_list.append(pid.view(-1).cpu())
            # feats_list.append(cross_x_bn.detach().cpu().float())
            feats_list.append(img_tokens)

        return torch.cat(pids_list, 0), torch.cat(feats_list, 0)

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        infer_prompt = getattr(self.args, "infer_prompt", "simplified").lower()
        fixed_text_feature = None
        if infer_prompt == "simplified":
            fixed_text_feature = self._build_fixed_text_feature(model, device)
            if fixed_text_feature is None:
                # fallback to composed if tokenizer unavailable
                infer_prompt = "composed"

        # ---- text queries from txt_loader ----
        qids_list, qfeats_list = [], []
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids_list.append(pid.view(-1).cpu())
            qfeats_list.append(text_feat.detach().cpu().float())
        qids = torch.cat(qids_list, 0)
        qfeats = torch.cat(qfeats_list, 0)

        # ---- image gallery from img_loader ----
        gids, gfeats = self._process_images(model, self.img_loader, infer_prompt, fixed_text_feature, device)

        return qfeats, gfeats, qids, gids

    def eval(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)

        # ensure both feature tensors are on the same device for matrix multiplication
        device = next(model.parameters()).device
        qfeats = qfeats.to(device)
        gfeats = gfeats.to(device)

        # compute similarity on device then move to CPU for numpy conversions later
        sims = (qfeats @ gfeats.t()).cpu()

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        rs = get_metrics(sims, qids, gids, "t2i", False)
        table.add_row(rs)

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True
            )
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(["i2t", i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP, i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9]])

        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.4f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.4f}"
        table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"

        self.logger.info("\n" + str(table))
        top1 = float(rs[1])
        self.logger.info("\n" + "best R1 = " + str(top1))
        return {
            "R1":   float(rs[1]),
            "R5":   float(rs[2]),
            "R10":  float(rs[3]),
            "mAP":  float(rs[4]),
            "mINP": float(rs[5]),
            "rSum": float(rs[6]),
        }