from interplm.sae.inference import load_sae_from_hf
from interplm.embedders.esm import ESM
import torch
import scipy

# get device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Get ESM embeddings for protein sequence
plm_layer = 33 # Choose ESM layer (1,9,18,24,30,33)
sequence = "MRWQEMGYIFYPRKLR"
emb_fpath_torch = './esm_emb.pt'
latent_fpath_torch = './sae_latent_temp.pt'
latent_sparse_fpath_npz = './sae_latent_temp.npz'

esm = ESM(model_name="facebook/esm2_t33_650M_UR50D")
embeddings = esm.embed_single_sequence(sequence=sequence, layer=plm_layer)
embeddings = torch.from_numpy(embeddings).float().to(device)
torch.save(embeddings, emb_fpath_torch)
print('embeddings:', type(embeddings), embeddings.shape)

# Load SAE model and extract features
sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=plm_layer).to(device)
features = sae.encode(embeddings)
torch.save(features, latent_fpath_torch)

# convert to sparse array in numpy
features = features.cpu().detach().numpy()
print('features:', type(features), features.shape)
features_sparse = scipy.sparse.csr_matrix(features)
scipy.sparse.save_npz(latent_sparse_fpath_npz, features_sparse)

# reload to check
features_reloaded = scipy.sparse.load_npz(latent_sparse_fpath_npz).toarray()
print(((features_sparse==features) == False).any())
