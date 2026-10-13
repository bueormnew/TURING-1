"""Demo rápida LFM2 denso (sin MoE): forward, conteo de parámetros y generación."""

import torch

from lfm2 import LFM2ForCausalLM, get_preset


def main() -> None:
    torch.manual_seed(0)
    for name in ["350m", "700m", "1.2b", "2.6b"]:
        cfg = get_preset(name)
        print(
            f"{name}: d={cfg.hidden_size} L={cfg.num_hidden_layers} "
            f"FF={cfg.effective_intermediate_size} "
            f"conv={cfg.num_conv_blocks} attn={cfg.num_attention_blocks} "
            f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}/{cfg.head_dim}"
        )

    # Modelo pequeño para CPU (misma arquitectura, dims reducidas).
    cfg = get_preset("350m")
    cfg.vocab_size = 1024
    cfg.hidden_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_hidden_layers = 6
    cfg.full_attn_idxs = [2, 5]
    cfg.layer_types = None
    cfg.__post_init__()
    cfg.block_ff_dim = 256
    model = LFM2ForCausalLM(cfg).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros demo: {total/1e6:.2f}M (atados: {cfg.tie_word_embeddings})")

    ids = torch.randint(0, 1024, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids)
        print("logits:", tuple(out['logits'].shape))
        gen = model.generate(ids, max_new_tokens=12, do_sample=True, temperature=0.7, top_p=0.9, min_p=0.05)
        print("generado:", tuple(gen.shape), gen[0, -12:].tolist())


if __name__ == "__main__":
    main()
