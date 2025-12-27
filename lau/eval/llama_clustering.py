import json
import torch
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from json_repair import repair_json

def initialize_llama_pipeline(model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Loads Llama 3.1 8B Instruct model using Hugging Face pipeline."""
    print(f"Loading model {model_id}...")
    
    # Use torch.bfloat16 for efficiency on modern GPUs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Automatically determines where to place layers (e.g., GPU)
    )
    
    # Create the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False, # Use deterministic decoding for consistent curation
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )
    print("Model loaded successfully.")
    return pipe, tokenizer

class ClassificationOutput(BaseModel):
    """Schema for the LLM-generated label for a single sentence."""
    cluster_label: str = Field(
        ...,
        description="The cluster name chosen from the predefined list that best describes the sentence."
    )

def create_sequential_labeling_prompt(
    sentence: str, 
    predefined_labels: List[str]
) -> List[Dict[str, str]]:
    """
    Constructs the prompt to ask the LLM to classify a single sentence 
    into one of the predefined cluster labels.
    """
    
    # Create a string list of labels for the prompt instructions
    label_list = ", ".join([f'"{label}"' for label in predefined_labels])
    json_schema_str = ClassificationOutput.model_json_schema()
    
    # --- SYSTEM INSTRUCTION (Role: Strict Classifier) ---
    system_instruction = (
        "You are a thematic classifier. Your task is to assign one cluster label to the provided sentence. "
        f"The label MUST be chosen ONLY from the following list of French labels: [{label_list}]. "
        "Your response MUST be the exact, case-sensitive label from this list. If the sentence is completely "
        "irrelevant or doesn't fit any category, you must choose 'AUTRE'. "
        "You MUST output a single, valid JSON object strictly adhering to the following schema."
        f"\n\nJSON SCHEMA: {json_schema_str}"
    )
    
    # --- USER PROMPT ---
    user_prompt = f"Sentence to classify: \"{sentence}\""

    # Llama 3 Chat Template structure
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages

def generate_labeled_manifest(
    manifest_path: str,
    output_path: str,
    predefined_labels: List[str],
    llama_pipe,
    llama_tokenizer
) -> List[Dict[str, Any]]:
    """
    Loads the manifest, sequentially asks the LLM to classify each French reference text, 
    and saves a new manifest with the cluster label appended.
    """
    labeled_samples = []
    
    # 1. Load the manifest data
    manifest_data = []
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                manifest_data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"❌ Error loading manifest file: {e}")
        return []

    num_samples = len(manifest_data)
    
    # --- Helper function for LLM inference (sequential) ---
    def get_llm_label(sentence: str) -> str:
        messages = create_sequential_labeling_prompt(sentence, predefined_labels)
        prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            outputs = llama_pipe(
                prompt,
                return_full_text=False,
                max_new_tokens=512, # Small output size is fast
                do_sample=False
            )
            raw_output = outputs[0]['generated_text'].strip().strip("```json").strip("```").strip()
            
            # FIX: Replace single backslashes with double backslashes
            raw_output = raw_output.replace('\\', '\\\\')

            # Parse and validate the JSON output
            # Attempt Standard JSON Parsing
            try:
                json_data = json.loads(raw_output)
            
            # 3. If standard parsing fails, fall back to robust repair
            except json.JSONDecodeError as e:
                print(f"  > ⚠️ JSON Decode Error for Q: {sentence[:30]}.... Attempting repair. Error: {e}")
                
                try:
                    # repair_json automatically fixes missing commas, unquoted keys, etc.
                    repaired_output = repair_json(raw_output)
                    json_data = json.loads(repaired_output)
                except Exception as repair_e:
                    print(f"  > ❌ Repair failed for Q: {sentence[:30]}.... Repair error: {repair_e}")
                    return "ERROR_PARSING"
            
            validated = ClassificationOutput.model_validate(json_data)
            return validated.cluster_label.strip()
            
        except Exception as e:
            # Handle JSON parse errors or validation failures
            print(f"  > ❌ Pipeline failure or unknown error: {e}")
            return "ERROR_PIPELINE"

    # --- 2. Main Iteration Loop ---
    for i, item in enumerate(manifest_data):
        sentence = item.get('french', None)
        if not sentence:
            print(f"Skipping sample {i}: Missing 'french' key.")
            continue
            
        print(f"Processing sample {i+1}/{num_samples}...")

        # Get the label from the LLM
        cluster_label = get_llm_label(sentence)
        
        # 3. Compile the new manifest item
        item['cluster_label'] = cluster_label
        
        labeled_samples.append(item)

    # 4. Save the final labeled manifest (JSONL format)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in labeled_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
        print(f"\n✅ Successfully labeled and saved {len(labeled_samples)} samples to {output_path}")
    except Exception as e:
        print(f"❌ Error saving labeled manifest: {e}")
        
    return labeled_samples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate responses using Llama 3.1 Instruct model.")
    parser.add_argument("manifest_path", type=str, help="Path to the manifest file with french refs.")
    parser.add_argument("output_path", type=str, help="Path to save the manifest file with clusters.")
    args = parser.parse_args()

    PREDEFINED_LABELS = [
    "La famille et les origines",
    "La société et la culture",
    "Les conflits et les rivalités",
    "La religion et la spiritualité",
    "La transmission des connaissances",
    "AUTRE" # Always include a fallback label
    ]

    llama, tokenizer = initialize_llama_pipeline()
    generate_labeled_manifest(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
        predefined_labels=PREDEFINED_LABELS,
        llama_pipe=llama,
        llama_tokenizer=tokenizer)
