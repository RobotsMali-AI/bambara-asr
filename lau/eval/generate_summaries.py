import os
import json
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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

def create_summarization_prompt(all_text: str) -> List[Dict[str, str]]:
    """
    Constructs the prompt for the LLM to summarize a large, potentially fragmented text.
    """
    # --- SYSTEM INSTRUCTION (New Role: Coherent Summarizer) ---
    system_instruction = (
        "You are an expert abstractive summarizer and thematic analyst. "
        "Your task is to take a collection of potentially fragmented and unrelated sentences, "
        "infer the most likely thematic connections or chronological sequences, "
        "and synthesize the entire collection into a single, cohesive, globally sensible summary. "
        "Your output must be **exactly 5 paragraphs** long and written in french. "
        "Each paragraph must cover a distinct, inferred topic or sequence."
    )

    # --- USER PROMPT ---
    user_prompt = (
        "Here is the collected text, where each line represents a separate transcription. "
        "Identify the major themes present across all lines, group the relevant sentences by these themes, "
        "and write the final summary in **exactly 5 paragraphs** that flow logically."
        "\n\n--- INPUT TEXT START ---\n"
        f"{all_text}"
        "\n--- INPUT TEXT END ---\n"
    )

    # Llama 3 Chat Template structure
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]
    
    return messages

def generate_cohesive_summary(
    manifest_path: str,
    text_key: str,
    llama_pipe,
    llama_tokenizer,
    output_summary_path: str
) -> str:
    """
    Loads text from a manifest file using the specified key, joins it, 
    prompts the LLM to summarize, and saves the result to a text file.
    
    Args:
        manifest_path: Path to the JSONL file containing the text data.
        text_key: The key in the JSON line (e.g., 'french', 'lau-model-x', 'asr-mt-model-y').
        llama_pipe: The initialized Hugging Face pipeline object.
        llama_tokenizer: The initialized Llama 3 tokenizer object.
        output_summary_path: Path to save the final .txt summary.
    
    Returns:
        The generated summary string.
    """
    
    # 1. Load and prepare the massive input text
    all_texts = []
    try:
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Ensure the key exists before appending
                if text_key in data:
                    all_texts.append(data[text_key])
    except Exception as e:
        print(f"❌ Error loading manifest: {e}")
        return ""
    
    if not all_texts:
        print(f"⚠️ No text found with key '{text_key}' in the manifest.")
        return ""

    # Join all separate lines into one large text block, separated by newlines
    all_text_block = "\n".join(all_texts)
    
    # Check if the block is too long for the LLM's context window (e.g., Llama 3 8B is 8,192 tokens)
    # Tokenization check is usually needed here, but we proceed assuming the input fits for simplicity.
    print(f"Loaded {len(all_texts)} entries. Total input size (characters): {len(all_text_block)}")

    # 2. Construct the prompt
    messages = create_summarization_prompt(all_text_block)

    # Apply the chat template and generate the raw prompt string
    prompt = llama_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 3. Run the pipeline (inference)
    print("Beginning LLM summarization inference...")
    try:
        outputs = llama_pipe(
            prompt,
            return_full_text=False,
            # Set a high max_new_tokens to allow for a comprehensive 5-paragraph summary
            max_new_tokens=2048, 
            do_sample=False,
        )
        # The output is the generated summary text (no JSON formatting)
        final_summary = outputs[0]['generated_text'].strip()
        
    except Exception as e:
        print(f"❌ Error during pipeline inference: {e}")
        return ""

    # 4. Save the final summary to the output file
    try:
        with open(output_summary_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        print(f"✅ Summary for key '{text_key}' successfully saved to {output_summary_path}")
    except Exception as e:
        print(f"❌ Error saving summary file: {e}")

    return final_summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Cohesive Summaries from Manifest Texts using Llama 3.1")
    parser.add_argument("manifest_path", type=str, help="Path to the manifest JSONL file")

    args = parser.parse_args()

    # Initialize the Llama pipeline
    llama, llama_tokenizer = initialize_llama_pipeline()
    
    manifest_file_path = args.manifest_path
    manifest_data: List[Dict] = []
    with open(manifest_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            manifest_data.append(json.loads(line.strip()))
    
    EXCLUDE_KEYS = {'audio_filepath', 'duration', 'french', 'bam', 'ast-soloni-ctc', 'ast-soloni-tdt', 'asr-soloni-ctc', 'asr-soloni-tdt'}
    
    model_keys = [
        key for key in manifest_data[0].keys()
        if key not in EXCLUDE_KEYS
    ]
    
    for key in model_keys:
        output_path = f"summary_{key}.txt"
        print(f"\n--- Generating summary for key: {key} ---")
        # Generate the summary
        generate_cohesive_summary(
            manifest_path=manifest_file_path,
            text_key=key,
            llama_pipe=llama,
            llama_tokenizer=llama_tokenizer,
            output_summary_path=output_path
        )