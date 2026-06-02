from adapters import init
import adapters.composition as ac

from torch import save, load
from torch import clamp as t_clamp
from torch import nn
from torch import sum as t_sum
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class MoEGate(nn.Module):
    def __init__(self, hidden_size, num_classes, device):
        super(MoEGate, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device
        self.gate_linear_1 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.gate_linear_2 = nn.Linear(self.hidden_size, self.num_classes).to(device)
        
    def forward(self, gate_input):
        gate_embs_1 = F.relu(self.gate_linear_1(gate_input))
        gate_output = self.gate_linear_2(gate_embs_1)
        
        return gate_output
    
    def save(self, path):
        save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(load(path, map_location=self.device), strict=True)

class GatedCLS(nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer_name,
        num_classes,
        max_tokens=512,
        normalize=False,
        pooling_mode='mean',
        device='cpu',
    ):
        super(GatedCLS, self).__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.num_classes = num_classes
        
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        self.pooling_mode = pooling_mode

        self.device = device

        self._init_model()


    def _init_model(self) -> None:
        if 'cross' in self.model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.hidden_size = self.model.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if self.pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif self.pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif self.pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif self.pooling_mode == 'identity':
            self.pooling = self.identity

        self.cls = MoEGate(self.hidden_size, self.num_classes, self.device)

        
    def encoder(self, sentences):
        encoded_input = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        ).to(self.device)

        if 'cross' in self.model_name:
            embeddings = self.model.base_model(**encoded_input).pooler_output
            if self.normalize:
                return F.normalize(embeddings, dim=-1)
            
        embeddings = self.model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])
    
    def forward(self, data):
        embedding = self.encoder(data['text'])
        cls_output = self.cls(embedding)

        return {'gate_output': cls_output}

    def save(self, path):
        save(self.state_dict(), path)

    def save_cls(self, cls_path):
        self.cls.save(cls_path)

    def save_adapter(self, path, adapter_name):
        self.model.save_adapter(path, adapter_name)

    def load(self, path):
        self.load_state_dict(load(path, map_location=self.device), strict=False)

    def load_cls(self, cls_path):
        self.cls.load(cls_path)

    def load_adapter(self, path, adapter_name):
        init(self.model)
        self.model.load_adapter(path)
        self.model.set_active_adapters(adapter_name)
        self.model = self.model.to(self.device)


    def add_adapter(self, adapter_name, config):
        init(self.model)
        self.model.add_adapter(adapter_name, config=config, set_active=True)
        self.model.train_adapter(adapter_name) 
        self.model = self.model.to(self.device)
 
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


class GatedBiEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer_name,
        num_classes,
        max_tokens=512,
        normalize=False,
        pooling_mode='mean',
        device='cpu',
    ):
        super(GatedBiEncoder, self).__init__()
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.num_classes = num_classes
        
        assert pooling_mode in ['max', 'mean', 'cls', 'identity'], 'Only cls, identity, max and mean pooling allowed'
        self.pooling_mode = pooling_mode

        self.device = device
        
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.hidden_size = self.model.config.hidden_size
        
        if self.pooling_mode == 'mean':
            self.pooling = self.mean_pooling
        elif self.pooling_mode == 'max':
            self.pooling = self.max_pooling
        elif self.pooling_mode == 'cls':
            self.pooling = self.cls_pooling
        elif self.pooling_mode == 'identity':
            self.pooling = self.identity


        self.cls = MoEGate(self.hidden_size, self.num_classes, self.device)


    def get_domain(self, data):
        self.model.set_active_adapters('cls')
        encoded_input = self.tokenizer(
            data, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        ).to(self.device)
        embeddings = self.model(**encoded_input)
        
        embedding = self.pooling(embeddings, encoded_input['attention_mask'])
            
        if self.normalize:
            embedding = F.normalize(embedding, dim=-1)
            
                
        cls_output = self.cls(embedding)

        return cls_output

    def set_active_adapters(self, domain, weights=None, mode='top_k'):
        if mode == 'top_k':
            self.model.set_active_adapters(domain)
        if mode == 'weighted':
            self.model.active_adapters = ac.Average(["physics", "psychology", "political_science", "computer_science"], weights=weights)
        
    def forward(self, sentences):
        encoded_input = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        ).to(self.device) 
        embeddings = self.model(**encoded_input)
        if self.normalize:
            return F.normalize(self.pooling(embeddings, encoded_input['attention_mask']), dim=-1)
        return self.pooling(embeddings, encoded_input['attention_mask'])

    def save(self, path):
        save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(load(path, map_location=self.device), strict=False)

    def load_adapters(self, paths):
        init(self.model)
        for adapter_paths in paths:
            self.load_adapter(adapter_paths)
    
    def load_cls(self, cls_path):
        self.cls.load(cls_path)

    def load_adapter(self, path):
        init(self.model)
        self.model.load_adapter(path)
        self.model = self.model.to(self.device)


    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return t_sum(token_embeddings * input_mask_expanded, 1) / t_clamp(input_mask_expanded.sum(1), min=1e-9)


class GatedCrossEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer_name,
        num_classes,
        max_tokens=512,
        normalize=False,
        device='cpu',
    ):
        super(GatedCrossEncoder, self).__init__()
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.normalize = normalize
        self.num_classes = num_classes
        
        self.device = device
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.hidden_size = self.model.config.hidden_size


        self.cls = MoEGate(self.hidden_size, self.num_classes, self.device)


    def get_domain(self, data):
        self.model.set_active_adapters('cls')
        encoded_input = self.tokenizer(
            data, 
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        ).to(self.device)
        if 'cross' in self.model_name:
            embeddings = self.model.base_model(**encoded_input).pooler_output
        else:
            embeddings = self.model(**encoded_input)
        
        if self.normalize:
            embedding = F.normalize(embeddings, dim=-1)
            
                
        cls_output = self.cls(embedding)

        return cls_output

    def set_active_adapters(self, domain, weights=None, mode='top_k'):
        if mode == 'top_k':
            self.model.set_active_adapters(domain)
        if mode == 'weighted':
            self.model.active_adapters = ac.Average(["physics", "psychology", "political_science", "computer_science"], weights=weights)

    def classifier(self, query, documents):
        encoded_input = self.tokenizer(
            query, documents,
            padding=True, 
            truncation=True, 
            max_length=self.max_tokens, 
            return_tensors='pt'
        )
        encoded_input = encoded_input.to(self.device)
        return self.model(**encoded_input).logits
        
    def forward(self, query, sentences):
        return self.classifier([query for s in sentences], sentences)

    def save(self, path):
        save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(load(path, map_location=self.device), strict=False)

    def load_adapters(self, paths):
        init(self.model)
        for adapter_paths in paths:
            self.load_adapter(adapter_paths)
    
    def load_cls(self, cls_path):
        self.cls.load(cls_path)

    def load_adapter(self, path):
        init(self.model)
        self.model.load_adapter(path)
        self.model = self.model.to(self.device)