
#!/usr/bin/env python3
import argparse
from synth_gen.config import SynthConfig
from synth_gen.generator import SynthGenerator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="example_output")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cell_px", type=int, default=24)
    args = ap.parse_args()

    cfg = SynthConfig(out_dir=args.out_dir, n_samples=args.n, cell_px=args.cell_px)
    gen = SynthGenerator(cfg)
    res = gen.generate()
    print(f"Generated {res['count']} samples in {res['out_dir']}")

if __name__ == "__main__":
    main()
