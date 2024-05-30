import torch
import torch.nn as nn
from models.llm import load_LLaMa_model
from models.ocr import load_LayoutLMv3

class Llama2Embeddings(nn.Module):
    def __init__(self, embed_tokens):
        super(Llama2Embeddings, self).__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids, position_ids=None):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds


class Llama2WithoutEmbedding(nn.Module):
    def __init__(self, layers, norm):
        super(Llama2WithoutEmbedding, self).__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, inputs_embeds, positional_ids=None):
        hidden_states = inputs_embeds
        if positional_ids is None:
            positional_ids = torch.arange(0, inputs_embeds.size(1)).unsqueeze(0)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids=positional_ids)[0]
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Llama2CausalWithoutEmbedding(nn.Module):
    def __init__(self, model, lm_head):
        super(Llama2CausalWithoutEmbedding, self).__init__()
        self.model = model
        self.lm_head = lm_head
    def forward(self, input_embeds):
        hidden_states = self.lm_head(self.model(input_embeds))
        return hidden_states
    

class Projector(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=2048):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        return self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x.float()).float()).float()).float())).half()
    
class LayoutLLM(nn.Module):
    def __init__(self, dtm, llm, embed, visual_projector, text_projector, device="cuda:0"):
        super(LayoutLLM, self).__init__()
        self.dtm = dtm
        self.visual_projector = visual_projector
        self.text_projector = text_projector
        self.embed = embed
        self.llm = llm
        self.device = device

    def forward(self, question_ids,
                input_ids, bbox, attention_mask, pixel_values,
                image_token_len=197, **kwargs):
        input_ids.to(self.device)
        bbox.to(self.device)
        attention_mask.to(self.device)
        pixel_values.to(self.device)
        question_ids.to(self.device)
        last_state = self.dtm(input_ids=input_ids,
                              bbox=bbox,
                              attention_mask=attention_mask,
                              pixel_values=pixel_values).last_hidden_state.to(self.device)
        fv, ft = last_state[:,:image_token_len,:].to(self.device), last_state[:,image_token_len:,:].to(self.device)
        hv = self.visual_projector(fv).to(self.device)
        ht = self.text_projector(ft).to(self.device)
        text_embed = self.embed(question_ids).to(self.device)
        concat = torch.cat((hv.to(self.device),
                            ht.to(self.device),
                            text_embed.to(self.device)), dim=1).to(self.device)
        return self.llm(concat).to(self.device)

    def prepare_inputs_for_generation(self, question_ids, **dtm_encoding):
        pass

def get_model(llm_model_name, layoutlm_model_name="microsoft/layoutlmv3-base", apply_ocr=False, device="cuda", token=None):
    llm_pipe = load_LLaMa_model(llm_model_name, access_token=token)
    _, layoutlm_model = load_LayoutLMv3(layoutlm_model_name, apply_ocr=apply_ocr)
    llm_model = llm_pipe.model
    embedding_layer = Llama2Embeddings(llm_model.model.embed_tokens)
    model_wo_embedding = Llama2WithoutEmbedding(llm_model.model.layers, llm_model.model.norm)
    llm_backbone = Llama2CausalWithoutEmbedding(model_wo_embedding, llm_model.lm_head)
    vp = Projector(768, 4096).to(device)
    tp = Projector(768, 4096).to(device)
    layout_llm = LayoutLLM(layoutlm_model, llm_backbone, embedding_layer, vp, tp, device=device)
    return layout_llm, llm_pipe, layoutlm_model