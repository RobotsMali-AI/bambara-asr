from ..Rewards.reward_model import RewardModel, masked_mean_pooling
from ..Rewards.reward_config import RewardConfig
import torch.nn.functional as F

class CriticModel(RewardModel) :
    
    def __init__(self, config : RewardConfig):
        
        super().__init__(config)
        
        self.combined_dim = self.cfg.audio_conv_channels
        
        self.head = self.build_head(self.combined_dim)
        
    def forward(self, audio, audio_attention_mask, labels=None, reward=None, **kwargs):
        
        out = self.audio_encoder(audio)
        out_t = out.transpose(1,2)
        
        if audio_attention_mask.size(1) != out_t.size(1):
            audio_attention_mask = F.interpolate(
                audio_attention_mask.unsqueeze(1).float(),
                size=out_t.size(1),
                mode="nearest"
            ).squeeze(1)
            
        audio_enc = masked_mean_pooling(out_t, audio_attention_mask)
        
        pred = self.head(audio_enc).squeeze(-1)
        
        if reward is None :
            
            return pred
        
        return F.mse_loss(pred, reward)


        
        
        
        
