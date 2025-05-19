import torch
import torch.nn as nn
from transformers import BlipProcessor, BlipModel

class BLIPBERTModel(nn.Module):
  def __init__(self, hidden_dim, text_model_name='bert-base-uncased'):
    super().__init__()
    # Dealing with text
    self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
    self.bert = BertModel.from_pretrained(text_model_name)

    # Dealing with rsa
    self.rsa_encoder = nn.Linear(hidden_dim*3,  self.bert.config.hidden_size) # transform rsa sequences into Bert hidden size
    self.rsa_decoder = BertForMaskedLM.from_pretrained(text_model_name)
    
    # Fusing the two inputs for IT
    self.fusion_layer = nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size)
    self.output_layer = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)  # Latent space output
    
    def forward(self, descriptions, state_action_seq):
        # Process text
        text_tokens = self.tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True)
        text_embeds = self.shared_encoder(**text_tokens)

        # Process r-s-a sequences
        rsa_embeds = self.state_action_encoder(state_action_seq)


        
        text_embeds_1 = self.bert(**text_inputs_1).last_hidden_state[:, 0, :]
        text_embeds_2 = self.bert(**text_inputs_2).last_hidden_state[:, 0, :]
        state_action_embeds = self.state_action_encoder(state_action_seq)  # Process state-action sequence
        
        fused_representation = self.fusion_layer(torch.cat([text_embeds_1, text_embeds_2, state_action_embeds], dim=-1))
        latent_representation = self.output_layer(fused_representation)  # Output latent representation
        
        return latent_representation
    

# Replace BLIP's text embedding module
model.text_encoder.embeddings.word_embeddings = CustomTextEmbedding()




class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, 768)  # Match BLIP hidden size

    def forward(self, text):
        """
        Encode the task description into a 768-dimensional embedding.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_output = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use CLS token
        return self.projection(text_output)

BlipForConditionalGeneration(
  (vision_model): BlipVisionModel(
    (embeddings): BlipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (encoder): BlipEncoder(
      (layers): ModuleList(
        (0-11): 12 x BlipEncoderLayer(
          (self_attn): BlipAttention(
            (dropout): Dropout(p=0.0, inplace=False)
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (projection): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): BlipMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (text_decoder): BlipTextLMHeadModel(
    (bert): BlipTextModel(
      (embeddings): BlipTextEmbeddings(
        (word_embeddings): Embedding(30524, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (encoder): BlipTextEncoder(
        (layer): ModuleList(
          (0-11): 12 x BlipTextLayer(
            (attention): BlipTextAttention(
              (self): BlipTextSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (output): BlipTextSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (crossattention): BlipTextAttention(
              (self): BlipTextSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (output): BlipTextSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (intermediate): BlipTextIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BlipTextOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (cls): BlipTextOnlyMLMHead(
      (predictions): BlipTextLMPredictionHead(
        (transform): BlipTextPredictionHeadTransform(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (transform_act_fn): GELUActivation()
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
        (decoder): Linear(in_features=768, out_features=30524, bias=True)
      )
    )
  )
)

self.sar_model = BLIP.text_model # ITC + ITM
self.sar_
self.text_model = BERT # ITC + ITM
self.generation_model = BLIPForGeneration.text_model # DT




BlipModel(
  (text_model): BlipTextModel(
    (embeddings): BlipTextEmbeddings(
      (word_embeddings): Embedding(30524, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.0, inplace=False)
    )
    (encoder): BlipTextEncoder(
      (layer): ModuleList(
        (0-11): 12 x BlipTextLayer(
          (attention): BlipTextAttention(
            (self): BlipTextSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): BlipTextSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (crossattention): BlipTextAttention(
            (self): BlipTextSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
            (output): BlipTextSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (intermediate): BlipTextIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BlipTextOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
    (pooler): BlipTextPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (vision_model): BlipVisionModel(
    (embeddings): BlipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (encoder): BlipEncoder(
      (layers): ModuleList(
        (0-11): 12 x BlipEncoderLayer(
          (self_attn): BlipAttention(
            (dropout): Dropout(p=0.0, inplace=False)
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (projection): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): BlipMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (visual_projection): Linear(in_features=768, out_features=512, bias=False)
  (text_projection): Linear(in_features=768, out_features=512, bias=False)
)
