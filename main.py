import sys
import os

if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    def set_int_max_str_digits(maxdigits: int) -> None: pass
    sys.set_int_max_str_digits = set_int_max_str_digits

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" 

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LogitsProcessor, LogitsProcessorList
import torch
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

ALLOWED_TARGETS = [
    "graph", "flex", "morsynt", "semlex:minor:gramm",
    "semlex:minor:constrmorph", "semlex:minor:syn", 
    "semlex:minor:semsim", "semlex:major"
]

def normalize_true_label(label):
    label = str(label).strip()
    if label.startswith("plur-"):
        label = label.replace("plur-", "")
    if label in ALLOWED_TARGETS:
        return label
    for target in ALLOWED_TARGETS:
        if target in label:
            return target
    return "other"

class RestrictedTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Prevent anything else from being selected
        mask = torch.ones_like(scores) * float('-inf')
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        return scores + mask

class FewShotVariantDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
        # --- PASTE YOUR EXACT SYSTEM RULES, EXPLANATIONS, AND FEW SHOTS HERE ---
        self.system_and_examples = (
            "<s>[INST] Tu es un expert en philologie et en traitement automatique des langues pour l'ancien français. "
            "Ta tâche consiste à classifier des variantes textuelles (lieux variants) issues de manuscrits médiévaux. "
            "Voici les définitions et explications des catégories utilisables avec des exemples.\n" 
            "Dans chaque exemple, la barre verticale | sépare les variantes, et les sigles entre parenthèses sont ceux des manuscrits:\n"
            "- graph: variation graphique (ex. païs (AB)|pays (FH)).\n"
            "- flex: variation flexionnelle (ex. savomes (CB)|savon (H)|savons (P)).\n"
            "- morsynt: variation morphosyntaxique (ex. les despose (E)|le deposera (G)).\n"
            "- semlex:minor:gramm: écart sémantico-lexical faible s’expliquant par une variante de grammème (ex. cele (Ao)|l’ (Ez)).\n"
            "- semlex:minor:constrmorph: écart sémantico-lexical faible s’expliquant par la morphologie constructionnelle (mervoillent (Ao)|esmerveillent (Ez)).\n"
            "- semlex:minor:syn: écart sémantico-lexical faible s’expliquant par une relation de synonymie (ex. trichier (Ao)|decevoir (Ez)).\n"
            "- semlex:minor:semsim: écart sémantico-lexical faible s’expliquant par proximité sémantique (sergant (B)|archir (M)).\n"
            "- semlex:major: écart sémantico-lexical fort (ex. acorde (Ao)| discorde (Ez)).\n"            
            "Voici quelques exemples pour te guider :\n"
            "Exemple 1 : Variantes : \"oi (HVGP)|avoie (F)| ai (ASRM)\" -> Classification : morsynt\n"
            "Exemple 2 : Variantes : \"consoil (HM)|conseil (PG)|consel (FRA)\" -> Classification : graph\n"
            "Exemple 3 : Variantes : \"consel (HPFGRMA)|secors (V)|confort (S)\" -> Classification : semlex:minor:syn\n\n"
            "Consigne stricte : Réponds UNIQUEMENT avec l'un des 8 labels listés ci-dessus. Aucun commentaire.\n"
            "Classification : [/INST]" # Close instruction block before providing target data
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We append the target instance right after the rules block
        # Mistral will expect the very next generated token to fulfill the response loop
        target_instance = f" Variantes : \"{self.df.iloc[idx]['readings']}\" -> Classification :"
        return {"prompt": self.system_and_examples + target_instance}

# Initialize Model and Setup Tokens
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
model.config.pad_token_id = tokenizer.eos_token_id

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, clean_up_tokenization_spaces=False)

# Collect the exact tokens Mistral matches to your label strings
allowed_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ALLOWED_TARGETS]
logits_processors = LogitsProcessorList([RestrictedTokensProcessor(allowed_token_ids)])

# Data Processing Pipeline
df_variants = pd.read_csv("Yvain_reviewed_2026_ids.tsv", sep="\t")
dataset = FewShotVariantDataset(df_variants)

y_true = [normalize_true_label(lbl) for lbl in df_variants["type"]]
y_pred = []

print("Running evaluation on full prompt context + allowed logit masking...")
outputs = generator(
    KeyDataset(dataset, "prompt"),
    batch_size=32,
    return_full_text=False,
    generation_config=GenerationConfig(
        max_new_tokens=1, # The very first token generated will be locked to your 8 categories
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    ),
    logits_processor=logits_processors
)

for out in tqdm(outputs, total=len(df_variants)):
    pred_token = out[0]["generated_text"].strip()
    
    matched_target = "other"
    for target in ALLOWED_TARGETS:
        if target.startswith(pred_token) and len(pred_token) > 0:
            matched_target = target
            break
    y_pred.append(matched_target)

print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, zero_division=0))
