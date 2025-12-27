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

# --- 1. Define the LLM Answer Schema ---
class SimpleResponse(BaseModel):
    """Schema for the LLM-generated answer to a single question."""
    answer: str = Field(..., description="The direct, short, factual answer (in French) to the question based ONLY on the provided transcription text.")

# --- 2. Prompt Construction for Answerer LLM ---

def create_answerer_prompt(text_input: str, question: str) -> List[Dict[str, str]]:
    """
    Constructs the prompt to ask the LLM to answer a question based on a given text.
    """
    json_schema_str = SimpleResponse.model_json_schema()
    
    # --- SYSTEM INSTRUCTION (New Role: Factual Answerer) ---
    system_instruction = (
        "You are a factual linguistic assistant. Your task is to read a transcription text "
        "and provide a direct, concise answer (in French) to the user's question. "
        "Your answer MUST be based ONLY on the content of the provided transcription. "
        "If the information required to answer the question is not present or is ambiguous, "
        "you must respond with 'NE PEUT RÉPONDRE' (unable to Answered). "
        "You MUST output a single, valid JSON object strictly adhering to the following schema."
        f"\n\nJSON SCHEMA: {json.dumps(json_schema_str)}"
    )

    # --- USER PROMPT ---
    user_prompt = (
        f"Transcription Text:\n---\n{text_input}\n---\n\n"
        f"Question: {question}"
    )

    # Llama 3 Chat Template structure
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages

# Assuming you have llama_pipe, llama_tokenizer initialized
# and the global class definitions (e.g., CuratedOutputList)

def answer_questions_and_compile_results(
    curated_json_path: str, 
    manifest_path: str, 
    diff: str,
    llama_pipe, 
    llama_tokenizer,
    output_comparison_path: str
) -> List[Dict[str, Any]]:
    """
    Loads curated questions and model outputs, runs LLM task completion for lau and ASR+MT, 
    and compiles the final comparison JSON.
    """
    # 1. Load the Curated Questions/Tasks (Gold Standard)
    with open(curated_json_path, 'r', encoding='utf-8') as f:
        curated_data = json.load(f)['samples']
    
    # Convert list to a dictionary for fast lookup by audio_filepath
    curated_dict = {item['audio_filepath']: item for item in curated_data}

    # 2. Load the Model Outputs (lau and ASR+MT)
    manifest_data = {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            manifest_data[data['audio_filepath']] = data
    
    # 3. Setup the final compilation dictionary
    final_comparison_results = []
    
    # The models to evaluate (matching your manifest keys)
    # NOTE: Assuming 'lau-soloni-ctc' and 'asr-mt-soloni-ctc' are your primary comparison outputs.
    # Adjust these keys if you use different models (tdt/soloba).
    lau_CTC = f'lau-soloni-ctc-{diff}'
    lau_TDT = f'lau-soloni-tdt-{diff}'

    # --- Helper function for LLM inference (sequential) ---
    def get_llm_answer(text: str, question: str) -> str:
        messages = create_answerer_prompt(text, question)
        prompt = llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        try:
            # Run the pipeline (sequential inference)
            outputs = llama_pipe(
                prompt,
                return_full_text=False,
                max_new_tokens=128,
                do_sample=False
            )
            raw_output = outputs[0]['generated_text'].strip().strip("```json").strip("```").strip()
            raw_output = raw_output.replace('\\', '\\\\') # <--- ADD THIS LINE
            
            # Parse and validate the JSON output
            # 2. Attempt Standard JSON Parsing
            try:
                json_data = json.loads(raw_output)
            
            # 3. If standard parsing fails, fall back to robust repair
            except json.JSONDecodeError as e:
                print(f"  > ⚠️ JSON Decode Error for Q: {question[:30]}.... Attempting repair. Error: {e}")
                
                try:
                    # repair_json automatically fixes missing commas, unquoted keys, etc.
                    repaired_output = repair_json(raw_output)
                    json_data = json.loads(repaired_output)
                except Exception as repair_e:
                    print(f"  > ❌ Repair failed for Q: {question[:30]}.... Repair error: {repair_e}")
                    return "ERROR_PARSING"
                
            validated = SimpleResponse.model_validate(json_data)
            return validated.answer.strip()
        
        except Exception as e:
            # Catches pipeline or other non-JSON-parsing errors
            print(f"  > ❌ Pipeline failure or unknown error: {e}")
            return "ERROR_PIPELINE"
            
    # --- 4. Main Evaluation Loop ---
    for audio_path, curated_item in curated_dict.items():
        if audio_path not in manifest_data:
            print(f"Skipping {audio_path}: not found in manifest.")
            continue
            
        manifest_item = manifest_data[audio_path]
        
        ctc_output = manifest_item.get(lau_CTC, "N/A")
        tdt_output = manifest_item.get(lau_TDT, "N/A")
        
        # Prepare data structure for this audio file
        task_results = {
            "audio_filepath": audio_path,
            "tasks": []
        }

        # --- Sub-Loop: Evaluate all questions for this sample ---
        for task in curated_item['task_questions']:
            question = task['task_question']
            right_answer = task['answer']
            
            print(f"\nProcessing {audio_path}: Question: {question[:50]}...")

            # 1. Get lau Answer
            lau_ctc_answer = get_llm_answer(ctc_output, question)
            
            # 2. Get ASR+MT Answer
            lau_tdt_answer = get_llm_answer(tdt_output, question)
            
            # 3. Compile task result
            task_results['tasks'].append({
                "task_question": question,
                "right_answer": right_answer,
                "lau_ctc_answer": lau_ctc_answer,
                "lau_tdt_answer": lau_tdt_answer
            })
            
        final_comparison_results.append(task_results)

    # 5. Save the final JSON
    with open(output_comparison_path, 'w', encoding='utf-8') as f:
        json.dump(final_comparison_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Task Completion Evaluation complete. Results saved to {output_comparison_path}")
    
    return final_comparison_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate responses using Llama 3.1 Instruct model.")
    parser.add_argument("curated_json_path", type=str, help="Path to the curated JSON file with questions.")
    parser.add_argument("manifest_path", type=str, help="Path to the manifest file with model outputs.")
    parser.add_argument("diff", type=str, help="Distinguishing suffix for multiple lau entries.")
    parser.add_argument("output_comparison_path", type=str, help="Path to save the output comparison JSON.")
    args = parser.parse_args()

    # Initialize Llama pipeline
    llama, tokenizer = initialize_llama_pipeline()

    # Run the answer generation and compilation
    answer_questions_and_compile_results(
        curated_json_path=args.curated_json_path,
        manifest_path=args.manifest_path,
        diff=args.diff,
        llama_pipe=llama,
        llama_tokenizer=tokenizer,
        output_comparison_path=args.output_comparison_path
    )
