import json
import torch
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- 1. Define the Structured Output Schema (UPDATED) ---

# Schema for a single curated result (Keywords and Tasks)
class SingleCuratedResult(BaseModel):
    """Schema for the LLM-generated keywords and questions for a single sentence."""
    critical_keywords: List[str] = Field(
        ...,
        description="A list of 1 to 5 critical keywords (names, places, key verbs) in French from the reference text."
    )
    task_questions: List[Dict[str, str]] = Field(
        ...,
        description="A list of 1 to 2 simple, factual questions/instructions in French, each with a 'task_question' and its 'answer'."
    )

# Schema for the FINAL output: a list of SingleCuratedResult objects
class CuratedOutputList(BaseModel):
    """The complete manifest of all curated evaluation samples in a list."""
    # The list structure itself becomes the top-level output that the model must generate
    results: List[SingleCuratedResult] = Field(..., description="A list of curated results, one for each input sentence.")

# --- 2. Initialize Llama 3 Model ---
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

# --- 3. Prompt Construction with Few-Shot Examples (BASED ON YOUR INPUT) ---
def create_curation_prompt(reference_texts: list[str]) -> str:
    """
    Constructs the few-shot prompt using Llama 3.1's chat template structure 
    to process multiple sentences and output a single JSON array.
    """
    # Define the desired JSON schema as a string for inclusion in the prompt
    # NOTE: We use CuratedOutputList to get the schema for the array output.
    json_schema_str = CuratedOutputList.model_json_schema()
    
    # --- FEW-SHOT EXAMPLES (Using your format) ---
    few_shot_data = [
        {
            "critical_keywords": ["président", "Dakar", "accord commercial", "10 millions d'euros"],
            "task_questions": [
                {"task_question": "Où le président s'est-il rendu?", "answer": "Dakar"},
                {"task_question": "Cite trois nombres inférieur au montant de l'accord signé?", "answer": "100; 1000; 65847"}
            ]
        },
        {
            "critical_keywords": ["équipe", "match", "3-1", "but", "Diarra"],
            "task_questions": [
                {"task_question": "Qui a marqué le but de la victoire?", "answer": "Diarra"},
                {"task_question": "Quel était le score final du match?", "answer": "3-1"}
            ]
        }
    ]
    
    # --- USER INPUT SECTION ---
    sentences = "\n".join(reference_texts)
    
    # --- SYSTEM INSTRUCTION SECTION (Based on your input) ---
    # We must ensure the system instruction is a separate message in the Llama 3 template
    system_instruction = f"""
Cutting Knowledge Date: December 2023
Today Date: 25 November 2025
 
You are an expert french linguist, specializing in speech translation evaluation, you are training an AI system to recognize and correctly transcribe keywords from speech conversations. 
To evaluate your system you are prompting an LLM to answer/solve a set of questons/tasks that can be completed only with an acceptable text translation of the original speech.
Thus your first task is to analyze a French text and extract critical informational elements (keywords) and create a list of questions/tasks to be completed (in french).
You MUST output a single, valid JSON object that strictly adheres to the following schema: {json_schema_str}
Do not include any text, dialogue, or explanation outside of the JSON block.
 
In order to accomplish that task, the assistant provides you with useful, few-shot examples. 
The user attached the list of sentences for which you are expected to perform this task (one sentence per line). Treat each sentence as independent utterances; i.e DO NOT try to link them in your output or consider any of them has a relation with another
"""
    
    # The full prompt is structured as a list of dicts for the tokenizer to process
    # The Llama 3 tokenizer will handle inserting the <|start_header_id|> and <|eot_id|> tags.
    messages = [
        {"role": "system", "content": system_instruction},
        # Few-Shot User Turn (includes the two example sentences)
        {"role": "user", "content": f"Reference French texts:\nLe président a voyagé à Dakar pour signer un accord commercial de 10 millions d'euros.\nL'équipe a remporté le match 3-1 grâce à un but de dernière minute de Diarra."},
        # Few-Shot Assistant Turn (includes the single JSON output)
        {"role": "assistant", "content": json.dumps({"results": few_shot_data}, indent=4, ensure_ascii=False)},
        # Current Batch User Turn (includes the sentences for the current batch)
        {"role": "user", "content": f"Reference French texts:\n{sentences}"},
    ]

    # We return the list of message dicts, which is the input format for the tokenizer's chat template
    return messages