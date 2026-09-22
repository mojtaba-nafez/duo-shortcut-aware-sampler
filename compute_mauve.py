"""
Standalone MAUVE score calculator — no hydra config, no repo-specific imports.

Reference texts are pulled automatically from an OpenWebText held-out split,
following the nanoGPT/SEDD/MDLM-lineage convention:
    dataset["train"].train_test_split(test_size=0.0005, seed=2357)["test"]
then packed into ~seq_len-token chunks (EOS-joined docs, sliced into
non-overlapping windows) so the reference length distribution matches your
generated sequences. If your gidd config uses a different seed/test_size,
override --owt_seed / --owt_test_size / --owt_full_split.

Usage:
    python compute_mauve.py --generated_path gen.json --seq_len 512


python compute_mauve.py   --generated_path /idiap/temp/mnafez/research/discrete-diffusion-guidance/greedy-tail-ablation-mauve-result/udlm-clean-loss-term-time-dependent-v2-step-512-greedy-1.json \
   --owt_dat_path /idiap/temp/mnafez/research/duo/data/openwebtext-valid_validation_bs512_unwrapped.dat   --seq_len 512 --print_samples 1


python compute_mauve.py   --generated_path /idiap/temp/mnafez/research/discrete-diffusion-guidance/greedy-tail-ablation-mauve-result/udlm-original-step-512-greedy-1.json \
   --owt_dat_path /idiap/temp/mnafez/research/duo/data/openwebtext-valid_validation_bs512_unwrapped.dat   --seq_len 512 --print_samples 1



python compute_mauve.py   --generated_path /idiap/temp/mnafez/research/duo/greedy-tail-ablation-mauve-result/duo-5000-original-greedy-1.json \
   --seq_len 512 --print_samples 1


To use your own reference texts instead of auto-pulling OWT, pass
--reference_path some.json (same flexible JSON loader as generated_path).
If your generated JSON structure isn't recognized, edit only `extract_texts`.
"""
import argparse, json, os, re
import mauve
from transformers import AutoTokenizer


def extract_texts(json_path):
    """Pull a list[str] out of a JSON file. Edit this if your structure differs."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"{json_path}: empty list")
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            for key in ("text", "seq", "sample"):
                if key in data[0]:
                    return [d[key] for d in data]
        raise ValueError(f"{json_path}: unrecognized list element type {type(data[0])}")

    if isinstance(data, dict):
        for key in ("generated_seqs", "generated_texts", "texts", "text", "samples", "seqs"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if "per_sample" in data and isinstance(data["per_sample"], list):
            return [d["text"] for d in data["per_sample"]]
        raise ValueError(f"{json_path}: no known text field in keys {list(data.keys())} — edit extract_texts()")

    raise ValueError(f"{json_path}: unsupported top-level JSON type {type(data)}")


_SPECIAL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]+\|>|<pad>|<unk>|\[MASK\]|\[PAD\]")

def clean_text(text):
    """Strip literal special-token artifacts (<|endoftext|>, <pad>, [MASK], ...) that
    leak into decoded text and make it look nothing like real human text to MAUVE's
    featurizer. Add more patterns to _SPECIAL_TOKEN_RE if your tokenizer uses others."""
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def report_stats(texts, label):
    """Quick sanity numbers: are these texts a reasonable length, and how repetitive
    are they? A collapsed avg_words or high duplicate_ratio explains both a suspiciously
    fast run and a MAUVE score that doesn't match PPL/LLM-judge."""
    lengths = [len(t.split()) for t in texts]
    dup_ratio = 1 - len(set(texts)) / len(texts) if texts else 0
    print(f"[{label}] n={len(texts)} avg_words={sum(lengths)/max(len(lengths),1):.1f} "
          f"min={min(lengths) if lengths else 0} max={max(lengths) if lengths else 0} duplicate_ratio={dup_ratio:.3f}")


def load_openwebtext_val_texts(n_needed, seq_len, tokenizer, dataset_name="Skylion007/openwebtext",
                                seed=2357, test_size=0.0005, streaming=True, buffer_size=10_000):
    """
    Build ~seq_len-token human reference texts from an OWT held-out split, packed the
    way SEDD/MDLM-lineage pretraining pipelines do: concatenate tokenized docs with an
    EOS between them, then slice into non-overlapping seq_len-token windows.

    streaming=True (default) avoids downloading the full ~54GB corpus but only
    approximates the held-out split via a shuffle-buffer, so it isn't guaranteed to
    exactly match the train/val boundary. Pass streaming=False (--owt_full_split) for
    the exact `train_test_split(test_size, seed)` reproduction if you have the disk/time.
    """
    from datasets import load_dataset

    if streaming:
        ds = load_dataset(dataset_name, split="train", streaming=True).shuffle(seed=seed, buffer_size=buffer_size)
    else:
        full = load_dataset(dataset_name, split="train")
        ds = full.train_test_split(test_size=test_size, seed=seed, shuffle=True)["test"]

    eos_id = tokenizer.eos_token_id
    buf, chunks = [], []
    for ex in ds:
        buf.extend(tokenizer.encode(ex["text"], add_special_tokens=False) + [eos_id])
        while len(buf) >= seq_len and len(chunks) < n_needed:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(chunks) >= n_needed:
            break

    return tokenizer.batch_decode(chunks, skip_special_tokens=True)


def load_dat_reference_texts(dat_path, block_size, tokenizer, n_needed, wrapped=False):
    """
    Read human reference texts straight from a pre-tokenized nanoGPT/SEDD/MDLM/duo-lineage
    .dat file (np.memmap of uint16 token ids) already on disk - no download, no network.

    - unwrapped (default): file is already laid out as (num_examples, block_size) rows,
      one example per row - matches "*_unwrapped.dat" files like yours.
    - wrapped: file is one flat concatenated token stream; sliced here into non-overlapping
      block_size windows - matches "*_wrapped.dat" files.

    Verify block_size matches the filename (e.g. bs512) and sanity check with
    --print_samples before trusting it - a wrong dtype/shape/wrapped guess can silently
    decode to garbage rather than erroring.
    """
    import numpy as np
    arr = np.memmap(dat_path, dtype=np.uint16, mode="r")
    if wrapped:
        n_chunks = min(n_needed, len(arr) // block_size)
        chunks = arr[: n_chunks * block_size].reshape(n_chunks, block_size)
    else:
        assert len(arr) % block_size == 0, (
            f"{dat_path}: length {len(arr)} not divisible by block_size {block_size} "
            "- wrong --owt_dat_block_size, or did you mean --owt_dat_wrapped?"
        )
        chunks = arr.reshape(-1, block_size)[:n_needed]
    return tokenizer.batch_decode(chunks.tolist(), skip_special_tokens=True)


def load_hf_dataset_texts(path, n_needed, tokenizer):
    """
    Load reference texts from a directory saved via HF `datasets.Dataset.save_to_disk()` -
    recognized by a dataset_info.json/state.json alongside an .arrow file. This is what
    "*.dat" directories in this lineage (e.g. duo's data/ folder) usually turn out to be,
    despite the misleading extension - presumably exactly what get_dataloaders(...).dataset
    returns. Tries common token-id column names, decoding with `tokenizer`; falls back to
    a "text" column if present.
    """
    from datasets import load_from_disk
    ds = load_from_disk(path)
    print(f"{path}: columns = {ds.column_names}")

    for col in ("input_ids", "ids", "tokens"):
        if col in ds.column_names:
            return tokenizer.batch_decode(ds[col][:n_needed], skip_special_tokens=True)

    if "text" in ds.column_names:
        return ds["text"][:n_needed]

    raise ValueError(f"{path}: no recognized column among {ds.column_names} - edit load_hf_dataset_texts()")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generated_path", required=True)
    p.add_argument("--reference_path", default=None, help="Optional JSON of reference texts; omit to auto-pull from OpenWebText val")
    p.add_argument("--output_path", default=None)
    p.add_argument("--seq_len", type=int, default=512, help="Approx token length of your generated sequences; used to chunk the OWT reference to match")
    p.add_argument("--tokenizer_name", default="gpt2")
    p.add_argument("--owt_dataset_name", default="Skylion007/openwebtext")
    p.add_argument("--owt_seed", type=int, default=2357)
    p.add_argument("--owt_test_size", type=float, default=0.0005)
    p.add_argument("--owt_full_split", action="store_true", help="Materialize the full OWT train split and take the exact held-out slice, instead of streaming")
    p.add_argument("--owt_dat_path", default=None, help="Path to a pre-tokenized .dat file (e.g. openwebtext-valid_validation_bs512_unwrapped.dat) to use as the reference set instead of downloading anything from HF - takes priority over the OWT streaming/full_split paths")
    p.add_argument("--owt_dat_block_size", type=int, default=None, help="Block size the .dat file was prepared with (e.g. 512 for a _bs512_ file). Defaults to --seq_len")
    p.add_argument("--owt_dat_wrapped", action="store_true", help="Set if the .dat filename says wrapped rather than unwrapped")
    p.add_argument("--max_text_length", type=int, default=1024, help="Matches the kuleshov-group/remdm reference eval call. Only truncates texts longer than this - harmless to leave at 1024 even for shorter sequences")
    p.add_argument("--keep_special_tokens", action="store_true", help="Skip stripping <|endoftext|>-style tokens (matches remdm's raw-decode convention). Default is to strip them, since your OWT reference is already clean and an unstripped generated side creates a one-sided artifact")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--print_samples", type=int, default=0, help="Print this many cleaned paired ref/generated samples before scoring, for a sanity check")
    p.add_argument("--featurize_model_name", default="gpt2", help="LM used to embed texts for MAUVE. Library default is gpt2 (fast, weaker signal); most papers use gpt2-large (slower, stronger signal) — try both and compare")
    p.add_argument("--batch_size", type=int, default=1, help="Featurization batch size; raise if you have GPU headroom to speed things up")
    args = p.parse_args()

    generated_texts = extract_texts(args.generated_path)

    if args.reference_path:
        human_texts = extract_texts(args.reference_path)
    elif args.owt_dat_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if os.path.isdir(args.owt_dat_path):
            print(f"Reading {len(generated_texts)} reference chunks from HF dataset dir {args.owt_dat_path}...")
            human_texts = load_hf_dataset_texts(args.owt_dat_path, len(generated_texts), tokenizer)
        else:
            block_size = args.owt_dat_block_size or args.seq_len
            print(f"Reading {len(generated_texts)} reference chunks from raw file {args.owt_dat_path} (block_size={block_size}, wrapped={args.owt_dat_wrapped})...")
            human_texts = load_dat_reference_texts(
                args.owt_dat_path, block_size, tokenizer, len(generated_texts), wrapped=args.owt_dat_wrapped,
            )
    else:
        print(f"Pulling {len(generated_texts)} reference chunks (~{args.seq_len} tokens) from {args.owt_dataset_name} val split...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        human_texts = load_openwebtext_val_texts(
            len(generated_texts), args.seq_len, tokenizer,
            dataset_name=args.owt_dataset_name, seed=args.owt_seed,
            test_size=args.owt_test_size, streaming=not args.owt_full_split,
        )

    if not args.keep_special_tokens:
        generated_texts = [clean_text(t) for t in generated_texts]
        human_texts = [clean_text(t) for t in human_texts]

    n = min(len(generated_texts), len(human_texts))
    if len(generated_texts) != len(human_texts):
        print(f"Warning: mismatch (generated={len(generated_texts)}, reference={len(human_texts)}), truncating both to {n}")
    generated_texts, human_texts = generated_texts[:n], human_texts[:n]

    if n < 200:
        print(f"Warning: only {n} samples — MAUVE's k-means quantization is unstable below a few hundred, "
              f"and the paper uses ~5000. Treat scores from this run as rough at best.")

    if args.print_samples > 0:
        print("=== Sample sanity check (cleaned text passed to MAUVE) ===")
        for i in range(min(args.print_samples, n)):
            print(f"--- reference[{i}] ---\n{human_texts[i][:300]}\n")
            print(f"--- generated[{i}] ---\n{generated_texts[i][:300]}\n")
        print("=== end sample check ===")

    report_stats(human_texts, "reference")
    report_stats(generated_texts, "generated")

    print(f"Computing MAUVE over {n} samples using featurizer={args.featurize_model_name}, batch_size={args.batch_size}...")
    results = mauve.compute_mauve(p_text=human_texts, q_text=generated_texts, device_id=args.device_id,
                                   max_text_length=args.max_text_length, featurize_model_name=args.featurize_model_name,
                                   batch_size=args.batch_size, verbose=False)
    mauve_score = float(results.mauve)
    print("MAUVE score:", mauve_score)

    output_path = args.output_path or (os.path.splitext(args.generated_path)[0] + "_mauve.json")
    with open(output_path, "w") as f:
        json.dump({"mauve": mauve_score, "generated_path": args.generated_path,
                   "reference_path": args.reference_path or f"{args.owt_dataset_name}[val]",
                   "n_samples": n}, f, indent=4)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()