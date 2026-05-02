"""19_dpo_loss_derivation.py - Cap 27: DPO loss paso a paso para 1 triple.

Verifica que `dpo_loss` del modulo es coherente con calculo manual.
Al iniciar DPO con policy=ref=SFT, log-ratios son 0 y loss=-log(0.5).
"""
import torch
from _models import load_pretrained_mini_llama, compute_logp_response, dpo_loss
from _eval import build_char_maps

torch.manual_seed(1337)
text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

policy = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
ref    = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
for p in ref.parameters():
    p.requires_grad_(False)

prompt = "INSTR: reverse 'cat'\nRESP: "
chosen = "tac\n"
rejected = "CAT\n"
beta = 0.1

p_ids = torch.tensor([c2i[c] for c in prompt],   dtype=torch.long)
c_ids = torch.tensor([c2i[c] for c in chosen],   dtype=torch.long)
r_ids = torch.tensor([c2i[c] for c in rejected], dtype=torch.long)

print("=== DPO loss paso a paso ===\n")
print(f"Prompt:   {prompt!r}")
print(f"Chosen:   {chosen!r}")
print(f"Rejected: {rejected!r}")
print(f"Beta:     {beta}\n")

logp_pi_w  = compute_logp_response(policy, p_ids, c_ids)
logp_pi_l  = compute_logp_response(policy, p_ids, r_ids)
logp_ref_w = compute_logp_response(ref,    p_ids, c_ids)
logp_ref_l = compute_logp_response(ref,    p_ids, r_ids)

print(f"log pi_theta(y_w|x)  = {logp_pi_w.item():+.4f}")
print(f"log pi_theta(y_l|x)  = {logp_pi_l.item():+.4f}")
print(f"log pi_ref(y_w|x)    = {logp_ref_w.item():+.4f}")
print(f"log pi_ref(y_l|x)    = {logp_ref_l.item():+.4f}")

ratio_w = logp_pi_w - logp_ref_w
ratio_l = logp_pi_l - logp_ref_l
print(f"\nlog ratio chosen   = {ratio_w.item():+.4f}")
print(f"log ratio rejected = {ratio_l.item():+.4f}")

z = beta * (ratio_w - ratio_l)
loss_manual = -torch.nn.functional.logsigmoid(z)
print(f"\nbeta*(ratio_w - ratio_l) = {z.item():+.4f}")
print(f"loss_manual = -log sigmoid(z) = {loss_manual.item():.4f}")

loss_helper = dpo_loss(policy, ref, p_ids, c_ids, r_ids, beta=beta)
print(f"loss_helper                  = {loss_helper.item():.4f}")
assert abs(loss_manual.item() - loss_helper.item()) < 1e-4, "helper mismatch with manual"
print("\nOK: helper coincide con calculo manual.")
print("\nAl iniciar DPO desde SFT, policy=ref => ratios=0 => loss=-log(0.5)=0.6931.")
