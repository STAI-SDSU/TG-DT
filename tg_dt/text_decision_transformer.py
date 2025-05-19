# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import transformers

from transformers import BlipForQuestionAnswering, BlipProcessor, BlipConfig

# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->blip
def blip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

class RSAEmbeddings(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=None, max_ep_len=4096, projection_dim=768):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.fusion_layer = torch.nn.Linear(hidden_size, projection_dim)

        self.ln = nn.LayerNorm(projection_dim, eps=1e-12, elementwise_affine=True)

    def forward(self, states, actions, rewards, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]
    
        # Process r-s-a sequence data
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3)
        stacked_inputs = stacked_inputs.reshape(batch_size, 3*seq_length, self.hidden_size)
        # print("*****")

        # process the 
        stacked_inputs = self.ln(self.fusion_layer(stacked_inputs)) # batch_size, 3*seq_length, hidden_size

        return stacked_inputs

class TextDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, 
            max_length=None, max_ep_len=4096, action_tanh=True, device='cuda', blip_config_="Salesforce/blip-vqa-base", bert_config="Salesforce/blip-image-captioning-base", **kwargs):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        blip_config = BlipConfig.from_pretrained(blip_config_)
        BLIP = BlipForQuestionAnswering.from_pretrained(blip_config_)
        self.text_encoder = BLIP.text_encoder # BlipTextModel
        self.rsa_decoder = BLIP.text_decoder # BlipTextLMHeadModel: bert, cls, pooler
        self.hidden_size = blip_config.text_config.hidden_size
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        # Tokenizer for text, r-s-a sequences
        self.text_tokenizer = BlipProcessor.from_pretrained(bert_config)
        self.rsa_tokenizer = RSAEmbeddings(state_dim, act_dim, hidden_size, max_length, max_ep_len, blip_config.text_config.hidden_size) #BlipTextModel: embeddings, encode

        self.text_projection = nn.Linear(blip_config.text_config.hidden_size, blip_config.projection_dim, bias=False)
        self.rsa_projection = nn.Linear(blip_config.text_config.hidden_size, blip_config.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(blip_config.logit_scale_init_value))

        # image text matching head
        self.tbm_head = nn.Linear(blip_config.text_config.hidden_size, 2)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(blip_config.text_config.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(blip_config.text_config.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(blip_config.text_config.hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, text, device = "cuda", attention_mask=None, expert=False):
        # Takes text description and try to predict what the corresponding action. 
        '''expert should not be included only text should be included'''
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
    
        # Process text data with the encoder
        text_tokens = self.text_tokenizer(text=text, return_tensors='pt', padding=True).to(device)
        text_embeddings = self.text_encoder(**text_tokens)

        # process rsa data
        rsa_tokens = self.rsa_tokenizer(states, actions, rewards, returns_to_go, timesteps)
        rsa_tokens_copy = rsa_tokens.clone()
        rsa_embeddings = self.rsa_decoder(inputs_embeds=rsa_tokens_copy, attention_mask=stacked_attention_mask, output_hidden_states=True)
        # rsa_embeddings = self.rsa_decoder(inputs_embeds=rsa_tokens, attention_mask=stacked_attention_mask, output_hidden_states=True)

        # Compute different losses
        # TBC loss
        text_embeds = text_embeddings.last_hidden_state
        text_embeds = self.text_projection(text_embeds)[:, 0, :]

        rsa_embeds = rsa_embeddings.hidden_states[-1]
        rsa_embeds = self.rsa_projection(rsa_embeds)[:, 0, :]

        # print('shapes:', rsa_embeds.size(), text_embeds.size())

        # normalized features
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        rsa_embeds = rsa_embeds / rsa_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp().to(device=device)
        text_embeds = text_embeds.to(device=device, dtype=rsa_embeds.dtype)
        logits_per_rsa = torch.matmul(rsa_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_rsa.t()

        # import pdb
        # pdb.set_trace()
        # print("********(logits_per_text)********", logits_per_text.size())
        tbc_loss = blip_loss(logits_per_text)

        if expert: 
            labels = torch.ones(batch_size, device=device).long()
        else:
            labels = torch.zeros(batch_size, device=device).long()

        ca_outputs = self.rsa_decoder(inputs_embeds=rsa_tokens, attention_mask=stacked_attention_mask, encoder_hidden_states=text_embeddings[0],output_hidden_states=True)
        tbm_outputs = ca_outputs.hidden_states[-1]
        tbm_outputs = self.tbm_head(tbm_outputs[:, 0, :])
        tbm_loss = F.cross_entropy(tbm_outputs, labels)
            
        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        x = ca_outputs.hidden_states[-1]
        
        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state

        return state_preds, action_preds, return_preds, tbc_loss, tbm_loss

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, text, device = "cuda", attention_mask=None, expert=False, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # Note: prompt within kwargs
        _, action_preds, return_preds, _, _ = self.forward(
            states, actions, None, returns_to_go, timesteps, text, device=device, expert=expert, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]


