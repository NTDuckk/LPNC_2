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
    # Convert similarity to numpy for global argsort (descending)
    if torch.is_tensor(similarity):
        sim_np = similarity.detach().cpu().numpy()
    else:
        sim_np = np.array(similarity)

    # argsort descending across each row
    indices_np = np.argsort(-sim_np, axis=1)

    # ensure pids are numpy arrays
    q_pids_np = q_pids.detach().cpu().numpy() if torch.is_tensor(q_pids) else np.array(q_pids)
    g_pids_np = g_pids.detach().cpu().numpy() if torch.is_tensor(g_pids) else np.array(g_pids)

    pred_labels = g_pids_np[indices_np]
    matches = (pred_labels == q_pids_np[:, None])  # boolean numpy array

    num_q, num_g = matches.shape

    all_cmc_list = []
    all_AP = []
    all_INP = []
    num_valid_q = 0

    for i in range(num_q):
        match_row = matches[i]
        if not match_row.any():
            # skip queries that have no matching gallery
            continue

        # cumulative matches (binary)
        cmc = np.cumsum(match_row.astype(np.int32))
        cmc[cmc > 1] = 1
        all_cmc_list.append(cmc[:max_rank])
        num_valid_q += 1

        # AP computation (numpy) following eval_func
        num_rel = match_row.sum()
        tmp_cmc = np.cumsum(match_row.astype(np.float32))
        y = np.arange(1, tmp_cmc.shape[0] + 1).astype(np.float32)
        tmp = tmp_cmc / y
        tmp = tmp * match_row.astype(np.float32)
        AP = tmp.sum() / float(num_rel)
        all_AP.append(AP)

        # INP: precision at the last correct match
        last_pos = np.where(match_row)[0][-1]
        inp_val = tmp_cmc[last_pos] / float(last_pos + 1)
        all_INP.append(inp_val)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc_np = np.asarray(all_cmc_list).astype(np.float32)
    all_cmc = all_cmc_np.sum(0) / float(num_valid_q)
    mAP = float(np.mean(all_AP))
    mINP = float(np.mean(all_INP))

    # convert back to torch tensors to preserve previous interface
    all_cmc_t = torch.from_numpy(all_cmc)
    mAP_t = torch.tensor(mAP, dtype=torch.float32)
    mINP_t = torch.tensor(mINP, dtype=torch.float32)
    indices_t = torch.from_numpy(indices_np).long()

    return all_cmc_t, mAP_t, mINP_t, indices_t


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
                bsz = image_tokens.shape[0]

                if infer_prompt == "composed":
                    # S*_global from CLS token
                    i_feats = image_tokens[:, 0, :].float()   # (B, D)
                    # keep IM2TEXT input in float to match module parameter dtypes
                    s_global = model.img2text(i_feats)  # (B, D)

                    # S*_local from patch tokens (no t_feats refinement, no W)
                    patch_feats = image_tokens[:, 1:, :].float()  # (B, M, D)
                    s_local = model.local_pseudo(patch_feats)      # (B, D)

                    # S* = S*_global + S*_local (keep both in float dtype)
                    pseudo_token = s_global + s_local       # (B, D)

                    with autocast():
                        prompts = model.prompt_learner(pseudo_token)
                        text_feature = model.text_encoder(
                            prompts, model.prompt_learner.tokenized_prompts
                        )
                else:
                    # simplified: "A photo of a person" (no mapping network)
                    text_feature = fixed_text_feature.expand(bsz, -1)

                # cross-attn  (Q=text_feature, K/V=image_tokens)
                with autocast():
                    cross_x = model.cross_former(
                        text_feature.unsqueeze(1), image_tokens, image_tokens
                    )
                    cross_x_bn = model.bottleneck_proj(cross_x.squeeze(1))  # (B, D)

            pids_list.append(pid.view(-1).cpu())
            feats_list.append(cross_x_bn.detach().cpu().float())

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

        # ---- image-only evaluation: both query & gallery use the same pipeline ----
        gids, gfeats = self._process_images(model, self.img_loader, infer_prompt, fixed_text_feature, device)
        # query = gallery (no captions used)
        qids, qfeats = gids, gfeats

        return qfeats, gfeats, qids, gids

    def eval(self, model, i2t_metric=False):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)

        sims = qfeats @ gfeats.t()

        # query == gallery -> exclude self-matches (diagonal)
        sims.fill_diagonal_(-float('inf'))

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        rs = get_metrics(sims, qids, gids, "i2i", False)
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
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"

        self.logger.info("\n" + str(table))
        top1 = float(rs[1])
        self.logger.info("\n" + "best R1 = " + str(top1))
        return top1