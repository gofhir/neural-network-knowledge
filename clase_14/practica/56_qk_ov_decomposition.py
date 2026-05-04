"""56_qk_ov_decomposition.py - Cap 56: descomposicion QK y OV de la cabeza top."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import qk_circuit, ov_circuit

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

# Top previous-token head del cap 54: block.2 head.0
LAYER = 2
HEAD = 0
attn = model.blocks[LAYER].attn
d_k = attn.d_k  # 32
d_model = 128
group_size = attn.group_size  # 2
kv_group = HEAD // group_size  # head 0 -> kv group 0

# Extraer slices del peso para esta cabeza
# W_Q.weight: (h_q * d_k, d_model) -> seleccionamos rows [head*d_k : (head+1)*d_k]
# Convencion nn.Linear: out = x @ W.T, asi que W.weight: (out_features, in_features)
W_Q_full = attn.W_Q.weight  # (h_q * d_k = 128, d_model = 128)
W_K_full = attn.W_K.weight  # (h_kv * d_k = 64, d_model = 128)
W_V_full = attn.W_V.weight  # (h_kv * d_k = 64, d_model = 128)
W_O_full = attn.W_O.weight  # (d_model = 128, h_q * d_k = 128)

W_Q = W_Q_full[HEAD * d_k:(HEAD + 1) * d_k, :].T  # (d_model, d_k) = (128, 32)
W_K = W_K_full[kv_group * d_k:(kv_group + 1) * d_k, :].T  # (d_model, d_k)
W_V = W_V_full[kv_group * d_k:(kv_group + 1) * d_k, :].T  # (d_model, d_k)
W_O = W_O_full[:, HEAD * d_k:(HEAD + 1) * d_k].T  # (d_k, d_model)

print(f"=== Descomposicion de block.{LAYER} head.{HEAD} (top previous-token) ===\n")
print(f"d_model={d_model}, d_k={d_k}, kv_group={kv_group}")
print(f"W_Q shape: {tuple(W_Q.shape)}")
print(f"W_K shape: {tuple(W_K.shape)}")
print(f"W_V shape: {tuple(W_V.shape)}")
print(f"W_O shape: {tuple(W_O.shape)}\n")

# Computar circuitos
QK = qk_circuit(W_Q, W_K).cpu()  # (d_model, d_model)
OV = ov_circuit(W_V, W_O).cpu()  # (d_model, d_model)

print(f"QK circuit shape: {tuple(QK.shape)}")
print(f"OV circuit shape: {tuple(OV.shape)}\n")

# Estadisticas
print("=== Estadisticas del QK circuit ===")
print(f"||QK||_F (Frobenius)  = {QK.norm().item():.3f}")
print(f"Rank efectivo (>1e-4) = {(torch.linalg.svdvals(QK) > 1e-4).sum().item()}")
sv = torch.linalg.svdvals(QK)
print(f"Top-5 singular values = {sv[:5].tolist()}")
print(f"||QK||_F^2 / ||W_Q W_K^T||^2 = (decomposicion de bajo rango)")

print("\n=== Estadisticas del OV circuit ===")
print(f"||OV||_F (Frobenius)  = {OV.norm().item():.3f}")
print(f"Rank efectivo (>1e-4) = {(torch.linalg.svdvals(OV) > 1e-4).sum().item()}")
sv_ov = torch.linalg.svdvals(OV)
print(f"Top-5 singular values = {sv_ov[:5].tolist()}")

# Test funcional: ¿el OV circuit "copia" tokens?
# Un OV de copia ideal seria identidad. Medir cuanto se parece a I.
print("\n=== Test: ¿OV se parece a la identidad? (copy circuit test) ===")
I = torch.eye(d_model)
diff = (OV - I).norm().item()
ratio = diff / I.norm().item()
print(f"||OV - I||_F / ||I||_F = {ratio:.3f}")
print(f"Si ~0: OV es matriz identidad (copy puro)")
print(f"Si ~1: OV difiere completamente de identidad")

# Aplicar QK y OV a embeddings reales
# E -> QK -> E_T da matriz de 'preferencia de query token X por key token Y'
print("\n=== QK aplicado a embeddings: ¿que tokens prefiere atender esta cabeza? ===")
E = model.tok_emb.weight.detach().cpu()  # (vocab, d_model)
QK_emb = E @ QK @ E.T  # (vocab, vocab) — score de query=row, key=col

print("Top-5 pares (query, key) con mayor score (excluyendo diagonal):")
qk_emb_no_diag = QK_emb.clone()
qk_emb_no_diag.fill_diagonal_(float('-inf'))
flat_idx = qk_emb_no_diag.flatten().topk(5).indices
for fi in flat_idx.tolist():
    q_idx = fi // QK_emb.shape[1]
    k_idx = fi % QK_emb.shape[1]
    q_ch = tok.id_to_char[q_idx].replace('\n', '\\n')
    k_ch = tok.id_to_char[k_idx].replace('\n', '\\n')
    print(f"  query={q_ch!r:>5} -> key={k_ch!r:>5}  score={QK_emb[q_idx, k_idx].item():.3f}")

print("\n=== OV aplicado a embeddings: ¿que tokens copia esta cabeza? ===")
OV_emb = E @ OV @ E.T  # (vocab, vocab) — score de copy: input=row -> output=col
print("Top-5 pares (input_token -> escribe_token) con mayor score:")
flat_idx = OV_emb.flatten().topk(5).indices
for fi in flat_idx.tolist():
    in_idx = fi // OV_emb.shape[1]
    out_idx = fi % OV_emb.shape[1]
    in_ch = tok.id_to_char[in_idx].replace('\n', '\\n')
    out_ch = tok.id_to_char[out_idx].replace('\n', '\\n')
    print(f"  input={in_ch!r:>5} -> output={out_ch!r:>5}  score={OV_emb[in_idx, out_idx].item():.3f}")
