import sys

# 1. Provide the getter if missing
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300

# 2. Provide a type-perfect setter if missing to pass strict signature validation
if not hasattr(sys, 'set_int_max_str_digits'):
    def set_int_max_str_digits(maxdigits: int) -> None:
        pass  # Intentionally do nothing safely

    sys.set_int_max_str_digits = set_int_max_str_digits

import os
os.environ["TORCH_DYNAMO_DISABLE"] = "1"  # Désactive torch._dynamo
os.environ["TORCHDYNAMO_DISABLE"] = "1"   # Alternative pour certaines versions


#from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# --- Vérification des versions ---
print(f"PyTorch version: {torch.__version__}")
import transformers
print(f"Transformers version: {transformers.__version__}")


# Charger Mistral 7B (ou une version plus légère si GPU limité)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    clean_up_tokenization_spaces=False  # Suppresses the BPE tokenizer warning
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  # 'torch_dtype' is deprecated, updated to 'dtype'
    device_map="auto"     # Automatically handles GPU/CPU mapping
)

# Create generation pipeline with tokenizer configuration cleanly passed through
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    clean_up_tokenization_spaces=False  # This silences the BPE warning inside the pipeline
)

#TODO: on laisse tomber le contexte et les "nonsense" readings pour le moment
def classify_variant(readings):#, types_autorises):
    prompt_v1 = f"""
    Tu es un expert en philologie et linguistique historiques spécialisé dans l'ancien français.
    Voici un lieu variant extrait d'une collation de manuscrits 
    (les variantes sont séparées par une barre verticale | et les sigles des témoins donnés entre parenthèses):

    ---
    Variantes : "{readings}"
    ---

    Classifie ce lieu variant selon la typologie suivante :
    1. "graph" : variation graphique (ex. "païs (AB) | pays (FH)").
    2. "flex" : variation flexionnelle (ex. "savomes (CB) | savon (H) | savons (P)").
    3. "morsynt" : variation morphosyntaxique (ex. "les despose (E) | le deposera (G)")
    4. "semlex:minor:gramm" : écart sémantico-lexical faible s’expliquant par une variante de grammème ("cele (Ao) | l’ (Ez)").
    5. "semlex:minor:constrmorph": écart sémantico-lexical faible s’expliquant par la morphologie constructionnelle ("mervoillent (Ao) | esmerveillent (Ez)").
    6. "semlex:minor:syn": écart sémantico-lexical faible s’expliquant par une relation de synonymie ("trichier (Ao) | decevoir (Ez)").
    7. "semlex:minor:semsim": écart sémantico-lexical faible s’expliquant par des relations cognitives-associatives de proximité et de contiguïté sémantique ("sergant (B) | archir (M)").
    8. "semlex:major": écart sémantico-lexical fort ("acorde (Ao) | discorde (Ez)")
    
    Réponds **uniquement** avec le type le plus probable, sans justification.
    """

    # Prompt structured to force a base model to complete with the target token
    prompt = f"""Tu es un expert en philologie et linguistique historiques spécialisé dans l'ancien français.
    Voici une typologie stricte de classification :
    - "graph" : variation graphique (ex. "païs (AB) | pays (FH)").
    - "flex" : variation flexionnelle (ex. "savomes (CB) | savon (H) | savons (P)").
    - "morsynt" : variation morphosyntaxique (ex. "les despose (E) | le deposera (G)")
    - "semlex:minor:gramm" : écart sémantico-lexical faible s’expliquant par une variante de grammème ("cele (Ao) | l’ (Ez)").
    - "semlex:minor:constrmorph": écart sémantico-lexical faible s’expliquant par la morphologie constructionnelle ("mervoillent (Ao) | esmerveillent (Ez)").
    - "semlex:minor:syn" : écart sémantico-lexical faible s’expliquant par une relation de synonymie ("trichier (Ao) | decevoir (Ez)").
    - "semlex:minor:semsim" : écart sémantico-lexical faible s’expliquant par proximité sémantique ("sergant (B) | archir (M)").
    - "semlex:major" : écart sémantico-lexical fort ("acorde (Ao) | discorde (Ez)")

    Variantes à classifier : "{readings}"
    Type le plus probable : """

    # Generate response
    # Pass an actual GenerationConfig object instead of a raw dictionary
    gen_config = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    outputs = generator(
        prompt,
        return_full_text=False,
        generation_config=gen_config
    )

    # Extract only the newly generated text
    predicted_type = outputs[0]["generated_text"].strip().split("\n")[0].replace('"', '').strip()
    return predicted_type

df_variants = pd.read_csv("Yvain_reviewed_2026_ids.tsv", sep="\t")

# Target column names (Adjust "type" to your actual TSV column name)
true_column = "type" 

# Arrays to hold values
y_true = []
y_pred = []

# Toggle this to True to activate few-shot learning
FEW_SHOT_MODE = False 

print(f"Starting evaluation (Few-Shot Mode: {FEW_SHOT_MODE})...")

# Loop over rows with a progress bar
for index, row in tqdm(df_variants.iterrows(), total=len(df_variants)):
    true_label = str(row[true_column]).strip()
    predicted_label = classify_variant(row["readings"])#, use_few_shot=FEW_SHOT_MODE)
    
    y_true.append(true_label)
    y_pred.append(predicted_label)

# Print Metrics Report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, zero_division=0))

