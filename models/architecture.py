import torch
import torch.nn as nn

class RegularizedNeuMF(nn.Module):
    def __init__(self, num_patients, num_medicines, sbert_dim, layers=[256, 128, 64], dropout_rate=0.4):
        super(RegularizedNeuMF, self).__init__()
        self.num_medicines = num_medicines
        
        self.patient_gmf = nn.Embedding(num_patients, layers[-1])
        self.med_gmf = nn.Embedding(num_medicines, layers[-1])
        
        self.patient_mlp = nn.Embedding(num_patients, layers[0] // 4) 
        self.med_mlp = nn.Embedding(num_medicines, layers[0] // 4)    
        
        mlp_modules = []
        input_dim = (layers[0] // 4) + (layers[0] // 4) + sbert_dim 
        
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(input_dim, layers[i+1]))
            mlp_modules.append(nn.BatchNorm1d(layers[i+1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout_rate))
            input_dim = layers[i+1]
            
        self.mlp_network = nn.Sequential(*mlp_modules)
        self.prediction_layer = nn.Linear(layers[-1] + layers[-1], num_medicines)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, patient_id, text_vectors):
        batch_size = patient_id.size(0)
        all_med_ids = torch.arange(self.num_medicines, device=patient_id.device, dtype=torch.long)
        
        all_med_gmf = self.med_gmf(all_med_ids).mean(dim=0).expand(batch_size, -1)
        all_med_mlp = self.med_mlp(all_med_ids).mean(dim=0).expand(batch_size, -1)
        
        gmf_p = self.patient_gmf(patient_id)
        gmf_v = torch.mul(gmf_p, all_med_gmf)
        gmf_v = torch.tanh(gmf_v)
        
        mlp_p = self.patient_mlp(patient_id)
        mlp_combined = torch.cat([mlp_p, all_med_mlp, text_vectors], dim=-1)
        mlp_v = self.mlp_network(mlp_combined)
        
        fusion = torch.cat([gmf_v, mlp_v], dim=-1)
        logits = self.prediction_layer(fusion)
        return self.sigmoid(logits)