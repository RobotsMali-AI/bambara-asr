import os
import json
import random
import argparse
import requests
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment
from bambara_normalizer import BambaraNumberNormalizer

# Initialize normalizers globally to avoid overhead
normalizer = BambaraNumberNormalizer()

def download_and_convert(url, output_path):
    """
    Downloads an audio file from a URL and converts it to mono channel.
    Returns the duration in seconds, or None if it fails.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Save temp downloaded file
        temp_path = output_path + ".temp"
        with open(temp_path, 'wb') as f:
            f.write(response.content)
            
        # Convert to Mono and get duration
        audio = AudioSegment.from_file(temp_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        audio.export(output_path, format="wav")
        duration = audio.duration_seconds
        
        # Cleanup temp file
        os.remove(temp_path)
        
        return round(duration, 3)
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None

def process_record(record, root_dir):
    """
    Processes a single JSON record.
    Returns a dictionary of manifest entries if successful.
    """
    results = {'main': None, 'amount': None, 'number': None}
    
    annotation = record.get('annotation')
    # Condition: verify that the annotation key isn't empty
    if not annotation:
        return results
        
    audio_id = record.get('audio_id', 'unknown_id')
    action = annotation.get('action', '')
    entities = annotation.get('entities', [])
    
    is_transfer = action in ["transfer_to_account", "transfer_to_momo"]
    
    # --- 1. Main Audio Processing ---
    main_url = record.get('audio_url')
    if main_url:
        filename = f"{audio_id}.wav"
        # Using forward slashes for cross-platform JSON manifest compliance
        rel_path = f"{root_dir}/audios/{filename}" 
        abs_path = os.path.join(root_dir, 'audios', filename)
        
        duration = download_and_convert(main_url, abs_path)
        if duration is not None:
            manifest_annot = {
                'scenario': annotation.get('scenario'),
                'action': action,
                # Special Exception: empty entities for transfer actions in main manifest
                'entities': [] if is_transfer else entities
            }
            results['main'] = {
                "audio_filepath": rel_path,
                "offset": 0,
                "duration": duration,
                "text": str(manifest_annot) # Cast dict to string using single quotes
            }

    # --- 2. Amount and Number Audio Processing (Transfer Actions Only) ---
    if is_transfer:
        # Extract fillers
        amount_filler = next((e.get('filler', '') for e in entities if e.get('type') == 'amount'), "")
        number_filler = next((e.get('filler', '') for e in entities if e.get('type') in ['account_number', 'phone_number']), "")
        
        amount_url = record.get('amount_audio_url')
        number_url = record.get('number_audio_url')
        
        # Process Amount
        if amount_url and amount_filler:
            amount_filename = f"amount_{audio_id}.wav"
            rel_amount_path = f"{root_dir}/amount/audios/{amount_filename}"
            abs_amount_path = os.path.join(root_dir, 'amount', 'audios', amount_filename)
            
            amt_duration = download_and_convert(amount_url, abs_amount_path)
            if amt_duration is not None:
                try:
                    normalized_amt = normalizer(amount_filler, is_money=True).replace("-", " ")
                    results['amount'] = {
                        "audio_filepath": rel_amount_path,
                        "duration": amt_duration,
                        "text": normalized_amt
                    }
                except Exception as e:
                    print(f"Amount Normalizer Error on '{amount_filler}': {e}")

        # Process Number
        if number_url and number_filler:
            number_filename = f"number_{audio_id}.wav"
            rel_number_path = f"{root_dir}/number/audios/{number_filename}"
            abs_number_path = os.path.join(root_dir, 'number', 'audios', number_filename)
            
            num_duration = download_and_convert(number_url, abs_number_path)
            if num_duration is not None:
                try:
                    normalized_num = normalizer(number_filler, is_money=False).replace("-", " ")
                    results['number'] = {
                        "audio_filepath": rel_number_path,
                        "duration": num_duration,
                        "text": normalized_num
                    }
                except Exception as e:
                    print(f"Number Normalizer Error on '{number_filler}': {e}")

    return results

def split_and_save(data_list, test_val, root_dir, prefix=""):
    """
    Shuffles data, applies the test split (ratio or absolute count), and writes to JSONL.
    """
    if not data_list:
        return

    random.shuffle(data_list)
    
    if isinstance(test_val, float) and 0.0 <= test_val <= 1.0:
        test_size = int(len(data_list) * test_val)
    else:
        test_size = int(test_val)
        
    test_size = min(test_size, len(data_list))
    
    test_data = data_list[:test_size]
    train_data = data_list[test_size:]
    
    # Determine save directory based on prefix
    save_dir = os.path.join(root_dir, prefix.replace("_", "")) if prefix else root_dir
    
    train_path = os.path.join(save_dir, 'train.jsonl')
    test_path = os.path.join(save_dir, 'test.jsonl')
    
    # Write Train
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # Write Test
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"[{prefix.upper() or 'MAIN'}] Saved {len(train_data)} train and {len(test_data)} test records.")

def main():
    parser = argparse.ArgumentParser(description="Process Firebase Audio JSON and generate manifests.")
    parser.add_argument("json_file", help="Path to the exported JSON file")
    parser.add_argument("root_dir", help="Root directory for outputs (e.g., 'ROOT')")
    parser.add_argument("--main_test", type=float, default=0.2, help="Ratio (0.0-1.0) or int for main test size")
    parser.add_argument("--amount_test", type=float, default=0.2, help="Ratio (0.0-1.0) or int for amount test size")
    parser.add_argument("--number_test", type=float, default=0.2, help="Ratio (0.0-1.0) or int for number test size")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent download workers")
    args = parser.parse_args()

    # Create Directories
    os.makedirs(os.path.join(args.root_dir, 'audios'), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, 'amount', 'audios'), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, 'number', 'audios'), exist_ok=True)

    # Load JSON Data
    with open(args.json_file, 'r', encoding='utf-8') as f:
        records = json.load(f)

    main_manifests = []
    amount_manifests = []
    number_manifests = []

    print(f"Starting processing of {len(records)} records using {args.workers} workers...")

    # Execute in Parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_record, record, args.root_dir): record for record in records}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res['main']: main_manifests.append(res['main'])
            if res['amount']: amount_manifests.append(res['amount'])
            if res['number']: number_manifests.append(res['number'])

    print("Processing complete. Generating splits...")

    # Apply splits and save
    split_and_save(main_manifests, args.main_test, args.root_dir, prefix="")
    split_and_save(amount_manifests, args.amount_test, args.root_dir, prefix="amount_")
    split_and_save(number_manifests, args.number_test, args.root_dir, prefix="number_")

if __name__ == "__main__":
    main()