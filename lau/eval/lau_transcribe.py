import argparse
import json

from hybrid_rnnt_ctc_lau_models import HybridRNNTCTClauModel

def transcribe(model_id: str, audios: list[str] | str, batch_size = 64, decoder:str = None) -> list[str]:
    hyps = []
    model = HybridRNNTCTClauModel.restore_from(model_id) # Use restore from if you're using one of best-soloni or best-quartznet model, you cannot load those with the same function
    model.eval()
    if "soloni" in model_id and decoder is None:
        print("Using default TDT decoder")

    if "soloni" in model_id and decoder is not None:
        if decoder == "ctc":
            ctc_decoding_cfg = model.cfg.aux_ctc.decoding
            model.change_decoding_strategy(decoder_type='ctc', decoding_cfg=ctc_decoding_cfg)
        elif decoder == "tdt" or decoder == "rnnt":
            decoding_cfg = model.cfg.decoding
            model.change_decoding_strategy(decoder_type='rnnt', decoding_cfg=decoding_cfg)
        else:
            raise ValueError(f"Invalid decoder type: {decoder}")
    
    out = model.transcribe(audios, batch_size=batch_size)
    for hyp in out:
        hyps.append(hyp.text)

    return hyps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using Hybrid RNNT-CTC lau Model")
    parser.add_argument("model_id", type=str, help="Path to the pre-trained model")
    parser.add_argument("manifest_path", type=str, help="Path to the manifest file containing audio file paths")
    parser.add_argument("diff", type=str, help="Distinguishing suffix for multiple lau entries")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite the output file changing existing key names")

    args = parser.parse_args()
    model_id = args.model_id
    manifest_path = args.manifest_path
    rewrite = args.rewrite

    manifest_data: list[dict] = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            manifest_data.append(json.loads(line.strip()))
    if rewrite:
        new_manifest_data = []
        for entry in manifest_data:
            new_entry = entry.copy()
            new_entry["ast-soloni-ctc"] = new_entry.pop("lau-soloni-ctc")
            new_entry["ast-soloni-tdt"] = new_entry.pop("lau-soloni-tdt")
            new_entry.pop("lau-soloba-ctc")
            new_entry.pop("asr-soloba-ctc")
            new_entry.pop("asr-mt-soloba-ctc")
            new_manifest_data.append(new_entry)
        manifest_data = new_manifest_data

    for decoding in ["tdt", "ctc"]:
        results = transcribe(model_id, manifest_path, decoder=decoding)
        assert len(results) == len(manifest_data), "Number of transcriptions does not match number of entries in manifest"
        for i, entry in enumerate(manifest_data):
            key_name = f"lau-soloni-{decoding}-{args.diff}"
            entry[key_name] = results[i]

    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Transcriptions added to {manifest_path}")
