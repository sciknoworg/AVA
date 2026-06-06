import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingDPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) natively ported for sentence embeddings.
    Instead of generative token probabilities, it uses text similarity as the reward proxy.
    """
    def __init__(self, model, ref_model, beta=0.1):
        super().__init__()
        self.model = model
        self.ref_model = ref_model
        self.beta = beta

    def forward(self, sentence_features, labels):
        anchor_feat, pref_feat, rej_feat = sentence_features

        policy_anchor = self.model(anchor_feat)['sentence_embedding']
        policy_pref = self.model(pref_feat)['sentence_embedding']
        policy_rej = self.model(rej_feat)['sentence_embedding']

        with torch.no_grad():
            ref_anchor = self.ref_model(anchor_feat)['sentence_embedding']
            ref_pref = self.ref_model(pref_feat)['sentence_embedding']
            ref_rej = self.ref_model(rej_feat)['sentence_embedding']

        policy_sim_pref = F.cosine_similarity(policy_anchor, policy_pref)
        policy_sim_rej = F.cosine_similarity(policy_anchor, policy_rej)

        ref_sim_pref = F.cosine_similarity(ref_anchor, ref_pref)
        ref_sim_rej = F.cosine_similarity(ref_anchor, ref_rej)

        policy_margin = policy_sim_pref - policy_sim_rej
        ref_margin = ref_sim_pref - ref_sim_rej

        logits = policy_margin - ref_margin
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss

