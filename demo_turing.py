"""Demo TURING-1 v1: presets, memoria, forward+MTP y especulativo."""

import torch

from turing import TuringForCausalLM, get_preset
from turing.utils import QUANT_RECIPE, print_report


def main() -> None:
    torch.manual_seed(0)
    for name in ["350m", "700m", "1.2b", "2.6b"]:
        print_report(get_preset(name))
    print(QUANT_RECIPE)

    model = TuringForCausalLM(get_preset("tiny")).eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Params tiny (con MoE+shared+MTP): {total/1e6:.2f}M")

    ids = torch.randint(0, 200, (1, 12))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
        print(f"lm={out['lm_loss'].item():.3f} mtp={out['mtp_loss'].item():.3f}")
        g = model.generate(ids, max_new_tokens=8, do_sample=False)
        s = model.generate_speculative(ids, max_new_tokens=8, do_sample=False)
        print("greedy == especulativo:", bool(torch.equal(g, s)))


if __name__ == "__main__":
    main()
