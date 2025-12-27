import json
import csv
import re
from typing import Dict, List, Any

# --- Helper function for robust keyword checking ---
def check_keyword_present(keyword: str, text: str) -> bool:
    """
    Checks if a lowercase keyword is present in a lowercase text.
    Uses simple 'in' operation as requested for maximum simplicity and robustness.
    """
    return keyword.lower() in text.lower()

def evaluate_keyword_spotting(
    manifest_jsonl_path: str,
    keywords_json_path: str,
    output_csv_path: str
) -> str:
    """
    Calculates the F1-score for keyword spotting for all model outputs 
    in the manifest against the curated keywords.

    Args:
        manifest_jsonl_path: Path to the main manifest with model outputs.
        keywords_json_path: Path to the JSON file with critical keywords.
        output_csv_path: Path to save the final F1 scores CSV.

    Returns:
        The path to the generated CSV file.
    """
    
    # 1. Load Data
    
    # Load manifest and store it in a list
    manifest_data = []
    with open(manifest_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            manifest_data.append(json.loads(line.strip()))

    # Load keywords (which is a list of dictionaries per sample)
    with open(keywords_json_path, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)
    
    # Restructure keywords into a dictionary for quick lookup by audio_filepath
    keywords_dict = {}
    for item in keywords_data['samples']:
        # Extract the list of keyword strings from the dictionary structure
        keywords = item['critical_keywords']
        keywords_dict[item['audio_filepath']] = keywords

    # 2. Identify Model Keys
    
    # Define keys to ignore (metadata and ground truth)
    EXCLUDE_KEYS = {'audio_filepath', 'duration', 'french', 'bam','asr-soloni-ctc', 'asr-soloni-tdt'}
    
    # Get all keys present in the first sample that are not excluded
    if not manifest_data:
        print("❌ Manifest data is empty.")
        return ""
        
    model_keys = [
        key for key in manifest_data[0].keys() 
        if key not in EXCLUDE_KEYS
    ]
    
    # 3. Initialize Score Tracking
    
    # Initialize total TP, FP, FN counts for each model
    model_metrics = {
        key: {'TP': 0, 'FN': 0, 'TotalGT': 0} 
        for key in model_keys
    }
    
    # 4. Main Evaluation Loop
    
    for sample in manifest_data:
        audio_path = sample['audio_filepath']
        ground_truth_keywords = keywords_dict.get(audio_path, [])
        
        if not ground_truth_keywords:
            # Skip samples with no keywords defined
            continue
            
        # Total expected keywords for this sample (denominator for Recall)
        total_gt_keywords = len(ground_truth_keywords)

        for model_key in model_keys:
            model_text = str(sample.get(model_key, "")).strip()
            
            # 4a. Count True Positives (TP) and False Negatives (FN)
            # TP = Found in model AND expected in GT
            # FN = Expected in GT BUT NOT found in model
            
            for gt_keyword in ground_truth_keywords:
                found = check_keyword_present(gt_keyword, model_text)
                
                if found:
                    model_metrics[model_key]['TP'] += 1
                else:
                    model_metrics[model_key]['FN'] += 1
            
            model_metrics[model_key]['TotalGT'] += total_gt_keywords
    
    # 5. Final Calculation and CSV Output
    
    results = []
    
    for model_key, metrics in model_metrics.items():
        TP = metrics['TP']
        FN = metrics['FN']
        
        # Calculate Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        results.append({
            'Model': model_key,
            'Total Critical Keywords': metrics['TotalGT'],
            'True Positives (TP)': TP,
            'False Negatives (FN)': FN,
            'Recall': f"{recall:.4f}",
        })
        
    # Save the results to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n✅ Keyword Spotting F1 Evaluation complete. Results saved to {output_csv_path}")
    
    return output_csv_path