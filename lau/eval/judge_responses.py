import csv

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

# --- 1. Define the Binary Decision Schema ---
class BinaryJudgment(BaseModel):
    """Schema for the Judge LLM's binary decision."""
    decision: str = Field(..., description="The binary decision, must be either 'YES' (Correct/Satisfying) or 'NO' (Incorrect/Insufficient).")
    rationale: str = Field(..., description="A very brief, one-sentence French explanation for the decision, referencing the reference text.")

# --- 2. Prompt Construction for Judge LLM ---
def create_judge_prompt(
    reference_text: str, 
    task_question: str, 
    model_answer: str, 
    right_answer: str
) -> List[Dict[str, str]]:
    
    # Get the raw schema string
    json_schema_str = BinaryJudgment.model_json_schema()
    
    # --- REVISED SYSTEM INSTRUCTION ---
    # The LLM knows what a JSON Schema is. We tell it to output the JSON object directly.
    system_instruction = (
        "You are a strict evaluation judge. Your task is to determine if a 'Model Answer' is a correct and satisfying response "
        "to a 'Task/Question', based ONLY on the 'Reference Text'. "
        "The 'Data curator's Answer' is provided only for context on the required information, but the 'Model Answer' should not be penalized "
        "if it is semantically equivalent, more detailed, or phrased differently, as long as the core fact is supported by the 'Reference Text'."
        "You MUST output a single following the below schema; do NOT add any conversational text."
        f"\n\nJSON SCHEMA DEFINITION:\n{json_schema_str}\n" # Inject raw schema string
    )
    
    # --- USER PROMPT (Slightly adjusted for clarity) ---
    user_prompt = (
        f"Reference Text: \"{reference_text}\"\n"
        f"Task Question: \"{task_question}\"\n"
        f"Data Curator's Answer: \"{right_answer}\"\n"
        f"Model Answer (To be judged): \"{model_answer}\"\n\n"
        "Provide your YES/NO decision and rationale in the requested JSON format."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages

def evaluate_answers_with_judge(
    answers_json_path: str, 
    manifest_jsonl_path: str, 
    llama_pipe, 
    llama_tokenizer,
    output_judgments_path: str
) -> List[Dict[str, Any]]:
    """
    Uses the Judge LLM to evaluate the correctness of lau and ASR+MT answers 
    and saves the detailed judgments and summary statistics.
    """
    
    # 1. Load Data
    with open(answers_json_path, 'r', encoding='utf-8') as f:
        answers_data = json.load(f)
    
    # Load original manifest for reference text (O(n) for lookup)
    reference_dict = {}
    with open(manifest_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            reference_dict[data['audio_filepath']] = data.get('french', '') # Use 'french' key
    
    # 2. Statistics tracking
    stats = {
        'total_tasks': 0,
        'lau_ctc_correct': 0,
        'lau_tdt_correct': 0,
    }
    final_judgments = []

    # --- Helper function for Judge LLM inference (sequential) ---
    
    def get_judge_decision(ref_text: str, q: str, model_ans: str, right_ans: str) -> str:
        messages = create_judge_prompt(ref_text, q, model_ans, right_ans)
        prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            # 1. Run the pipeline
            outputs = llama_pipe(
                prompt,
                return_full_text=False,
                max_new_tokens=512,
                do_sample=False
            )
            raw_output = outputs[0]['generated_text'].strip().strip("```json").strip("```").strip()
            raw_output = raw_output.replace('\\', '\\\\') # Clean up escaped backslashes
    
            # 2. Attempt Standard JSON Parsing
            try:
                json_data = json.loads(raw_output)
            
            # 3. If standard parsing fails, fall back to robust repair
            except json.JSONDecodeError as e:
                print(f"  > ⚠️ JSON Decode Error for Q: {q[:30]}.... Attempting repair. Error: {e}")
                
                try:
                    # repair_json automatically fixes missing commas, unquoted keys, etc.
                    repaired_output = repair_json(raw_output)
                    json_data = json.loads(repaired_output)
                except Exception as repair_e:
                    print(f"  > ❌ Repair failed for Q: {q[:30]}.... Repair error: {repair_e}")
                    return "ERROR_PARSING"
            
            # 4. Validate and return the decision
            validated = BinaryJudgment.model_validate(json_data)
            return validated.decision.upper().strip()
            
        except Exception as e:
            # Catches pipeline or other non-JSON-parsing errors
            print(f"  > ❌ Pipeline failure or unknown error: {e}")
            return "ERROR_PIPELINE"
            
    # --- 3. Main Evaluation Loop ---
    for item in answers_data:
        audio_path = item['audio_filepath']
        ref_text = reference_dict.get(audio_path, "Reference not found.")
        
        # Prepare the judgment result structure for this audio file
        judged_item = {
            "audio_filepath": audio_path,
            "reference_french": ref_text,
            "judgments": []
        }

        for task in item['tasks']:
            stats['total_tasks'] += 1
            
            q = task['task_question']
            right_ans = task['right_answer']
            lau_ctc_ans = task['lau_ctc_answer']
            lau_tdt_ans = task['lau_tdt_answer']
            
            print(f"\nJudging {audio_path} (Task {stats['total_tasks']}): {q[:40]}...")

            # 3a. Judge lau Answer
            lau_ctc_decision = get_judge_decision(ref_text, q, lau_ctc_ans, right_ans)
            if lau_ctc_decision.lower() == "yes":
                stats['lau_ctc_correct'] += 1

            # 3b. Judge ASR+MT Answer
            lau_tdt_decision = get_judge_decision(ref_text, q, lau_tdt_ans, right_ans)
            if lau_tdt_decision.lower() == "yes":
                stats['lau_tdt_correct'] += 1
            
            # Record the full result for the detailed JSON output
            judged_item['judgments'].append({
                "task_question": q,
                "right_answer": right_ans,
                "lau_ctc_answer": lau_ctc_ans,
                "lau_ctc_correct": lau_ctc_decision,
                "lau_tdt_answer": lau_tdt_ans,
                "lau_tdt_correct": lau_tdt_decision,
            })
            
        final_judgments.append(judged_item)

    # 4. Save Detailed Judgments JSON
    with open(output_judgments_path, 'w', encoding='utf-8') as f:
        json.dump(final_judgments, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Detailed LLM Judgments saved to {output_judgments_path}")

    # 5. Calculate and Save Statistics CSV
    
    # Calculate percentages
    if stats['total_tasks'] > 0:
        lau_ctc_accuracy = (stats['lau_ctc_correct'] / stats['total_tasks']) * 100
        lau_tdt_accuracy = (stats['lau_tdt_correct'] / stats['total_tasks']) * 100
    else:
        lau_ctc_accuracy = lau_tdt_accuracy = 0.0

    stats_data = [
        {"Metric": "Total Tasks Judged", "Value": stats['total_tasks'], "Unit": "Tasks"},
        {"Metric": "lau CTC Correct Count", "Value": stats['lau_ctc_correct'], "Unit": "Count"},
        {"Metric": "lau TDT Correct Count", "Value": stats['lau_tdt_correct'], "Unit": "Count"},
        {"Metric": "lau CTC Accuracy", "Value": lau_ctc_accuracy, "Unit": "%"},
        {"Metric": "lau TDT Accuracy", "Value": lau_tdt_accuracy, "Unit": "%"},
    ]
    output_stats_path = output_judgments_path.replace(".json", ".csv")
    with open(output_stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats_data[0].keys())
        writer.writeheader()
        writer.writerows(stats_data)
        
    print(f"✅ Summary Statistics saved to {output_stats_path}")
    
    return stats

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate lau and ASR+MT answers using Judge LLM")
    parser.add_argument("answers_json_path", type=str, help="Path to the JSON file with model answers")
    parser.add_argument("manifest_jsonl_path", type=str, help="Path to the original manifest JSONL file")
    parser.add_argument("output_judgments_path", type=str, help="Path to save the detailed judgments JSON")

    args = parser.parse_args()

    # Initialize Llama pipeline
    llama, llama_tokenizer = initialize_llama_pipeline()

    # Run the evaluation
    evaluate_answers_with_judge(
        answers_json_path=args.answers_json_path,
        manifest_jsonl_path=args.manifest_jsonl_path,
        llama_pipe=llama,
        llama_tokenizer=llama_tokenizer,
        output_judgments_path=args.output_judgments_path
    )
