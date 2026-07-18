import sys
import os

# Monkey patches for strict environment
if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    def set_int_max_str_digits(maxdigits: int) -> None: pass
    sys.set_int_max_str_digits = set_int_max_str_digits

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" 

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load Model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
# FIX: Set the pad token to the end-of-sequence token so batching works
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
# Ensure the model config matches the tokenizer pad configuration
model.config.pad_token_id = tokenizer.eos_token_id


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    clean_up_tokenization_spaces=False
)
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import Dataset

# 1. Custom lightweight Dataset to cleanly yield prompts in real-time
class VariantDataset(Dataset):
    def __init__(self, df, use_few_shot=False):
        self.df = df
        self.use_few_shot = use_few_shot
        
        # Pre-build our constant instruction strings
        self.base_prompt = """Tu es un expert en philologie et linguistique historiques spécialisé dans l'ancien français.
Voici une typologie stricte de classification :
- "graph" : variation graphique (ex. "païs (AB) | pays (FH)").
- "flex" : variation flexionnelle (ex. "savomes (CB) | savon (H) | savons (P)").
- "morsynt" : variation morphosyntaxique (ex. "les despose (E) | le deposera (G)")
- "semlex:minor:gramm" : écart sémantico-lexical faible s’expliquant par une variante de grammème ("cele (Ao) | l’ (Ez)").
- "semlex:minor:constrmorph": écart sémantico-lexical faible s’expliquant par la morphologie constructionnelle ("mervoillent (Ao) | esmerveillent (Ez)").
- "semlex:minor:syn" : écart sémantico-lexical faible s’expliquant par une relation de synonymie ("trichier (Ao) | decevoir (Ez)").
- "semlex:minor:semsim" : écart sémantico-lexical faible s’expliquant par proximité sémantique ("sergant (B) | archir (M)").
- "semlex:major" : écart sémantico-lexical fort ("acorde (Ao) | discorde (Ez)")\n"""

        self.few_shot_examples = """
Exemple 1 :
Variantes à classifier : "chevalier (A) | chevaliers (B)"
Type le plus probable : flex

Exemple 2 :
Variantes à classifier : "aimer (Ao) | cherir (Ez)"
Type le plus probable : semlex:minor:syn

Exemple 3 :
Variantes à classifier : "li rois (C) | le roy (D)"
Type le plus probable : graph
"""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        readings = self.df.iloc[idx]["readings"]
        # Updated for Mistral Instruct formatting
        prompt = f"<s>[INST] {self.base_prompt}\n"
        if self.use_few_shot:
            prompt += f"{self.few_shot_examples}\n"
        prompt += f'Variantes à classifier : "{readings}" [/INST] Type le plus probable : '

        return {"prompt": prompt}


# --- Optimized Evaluation Loop ---

df_variants = pd.read_csv("Yvain_reviewed_2026_ids.tsv", sep="\t")
true_column = "type"
FEW_SHOT_MODE = False  # Toggle to True to benchmark your few-shot prompt

# Instantiate our streaming dataset
dataset = VariantDataset(df_variants, use_few_shot=FEW_SHOT_MODE)

# Extract ground truth cleanly using pandas
y_true = df_variants[true_column].astype(str).str.strip().tolist()
y_pred = []

# Generation settings
gen_config = GenerationConfig(
    max_new_tokens=10,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

print(f"Starting optimized batch evaluation (Few-Shot Mode: {FEW_SHOT_MODE})...")

# pipeline handles batching automatically when passed a dataset and a batch_size
# Adjust batch_size=16 or 32 depending on your GPU VRAM headroom
outputs = generator(
    KeyDataset(dataset, "prompt"),
    batch_size=16, 
    return_full_text=False,
    generation_config=gen_config
)

# Iterate through streaming generator outputs using tqdm
for out in tqdm(outputs, total=len(df_variants)):
    predicted_type = out[0]["generated_text"].strip().split("\n")[0].replace('"', '').strip()
    y_pred.append(predicted_type)

# Print Metrics Report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, zero_division=0))
