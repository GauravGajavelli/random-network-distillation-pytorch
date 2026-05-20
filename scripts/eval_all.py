"""Runs all three experiment verifiers and prints a summary table.

Expects a layout under --runs-dir of per-experiment TensorBoard log directories:

  runs/
    exp1_lava_vanilla/
    exp1_lava_option_b/
    exp2_doorkey_ppo/
    exp2_doorkey_vanilla_rnd/
    exp2_doorkey_option_b/
    exp3_keycorridor_baseline/
    exp3_keycorridor_dsc/
    exp3_keycorridor_dsc_tv_off/

Exit code 0 only if all three experiments PASS.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_pitfall import evaluate as eval_pitfall
from eval_gravitar import evaluate as eval_gravitar
from eval_dsc import evaluate as eval_dsc


EXPECTED = {
    "pitfall_vanilla": "exp1_lava_vanilla",
    "pitfall_option_b": "exp1_lava_option_b",
    "gravitar_ppo": "exp2_doorkey_ppo",
    "gravitar_vanilla_rnd": "exp2_doorkey_vanilla_rnd",
    "gravitar_option_b": "exp2_doorkey_option_b",
    "dsc_baseline": "exp3_keycorridor_baseline",
    "dsc": "exp3_keycorridor_dsc",
    "dsc_typed": "exp3_keycorridor_dsc_typed",
    "dsc_tv_off": "exp3_keycorridor_dsc_tv_off",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs",
                   help="root directory containing per-experiment log dirs")
    p.add_argument("--figures-dir", default="scripts/figures")
    args = p.parse_args()

    root = Path(args.runs_dir)
    paths = {k: root / v for k, v in EXPECTED.items()}
    missing = [k for k, v in paths.items() if not v.exists()]

    # Tolerate missing optional dirs (dsc_tv_off and dsc_typed are optional).
    required = [k for k in missing if k not in ("dsc_tv_off", "dsc_typed")]
    if required:
        print(f"missing required run dirs: {required}")
        print("expected layout under --runs-dir:")
        for k, v in paths.items():
            marker = "  (optional)" if k == "dsc_tv_off" else ""
            print(f"  {k} -> {v}{marker}")
        sys.exit(2)

    print("=" * 60)
    print("Experiment 1: Pitfall proxy (LavaCrossing)")
    print("=" * 60)
    e1 = eval_pitfall(str(paths["pitfall_vanilla"]),
                      str(paths["pitfall_option_b"]))

    print()
    print("=" * 60)
    print("Experiment 2: Gravitar proxy (DoorKey-8x8)")
    print("=" * 60)
    e2 = eval_gravitar(str(paths["gravitar_ppo"]),
                       str(paths["gravitar_vanilla_rnd"]),
                       str(paths["gravitar_option_b"]),
                       figures_dir=args.figures_dir)

    print()
    print("=" * 60)
    print("Experiment 3a: DSC (point landmarks) vs Option B baseline")
    print("=" * 60)
    tv_off = str(paths["dsc_tv_off"]) if paths["dsc_tv_off"].exists() else None
    e3 = eval_dsc(str(paths["dsc_baseline"]),
                  str(paths["dsc"]),
                  dsc_tv_off_dir=tv_off,
                  figures_dir=args.figures_dir)

    e3_typed = None
    if paths["dsc_typed"].exists():
        print()
        print("=" * 60)
        print("Experiment 3b: DSC-typed (K-means clusters) vs point-DSC")
        print("=" * 60)
        e3_typed = eval_dsc(str(paths["dsc"]),
                            str(paths["dsc_typed"]),
                            figures_dir=args.figures_dir)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = [
        ("Pitfall proxy (Option B fix)", e1),
        ("Gravitar proxy (Option B fix)", e2),
        ("DSC (points) vs Option B baseline", e3),
    ]
    if e3_typed is not None:
        results.append(("DSC-typed vs point-DSC (type-learning contribution)", e3_typed))
    for name, code in results:
        status = "PASS" if code == 0 else ("FAIL" if code == 1 else "INCOMPLETE")
        print(f"  {name}: {status}")
    sys.exit(0 if all(c == 0 for _, c in results) else 1)


if __name__ == "__main__":
    main()
