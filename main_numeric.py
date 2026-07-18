import sys
import os

if not hasattr(sys, 'get_int_max_str_digits'):
    sys.get_int_max_str_digits = lambda: 4300
if not hasattr(sys, 'set_int_max_str_digits'):
    def set_int_max_str_digits(maxdigits: int) -> None: pass


    sys.set_int_max_str_digits = set_int_max_str_digits

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Optimize CUDA memory fragmentation to help stay within 24GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


# Restricts logit selection to exact tokens representing digits 1-8
class RestrictedDigitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.ones_like(scores) * float('-inf')
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        return scores + mask


class FewShotVariantDataset(Dataset):
    def __init__(self, df):
        self.df = df

        # --- Prompt Order Inversion & Formatting Optimization ---
        # Moving examples first, instructions last so the model doesn't drift
        self.prefix = (
            "<s>[INST] Tu es un expert en philologie et en traitement automatique des langues pour l'ancien français. "
            "Ta tâche consiste à classifier des variantes textuelles (lieux variants) issues de manuscrits médiévaux.\n\n"
            "Voici des exemples pour te guider :\n"
            "Variantes : \"oi (HVGP)|avoie (F)| ai (ASRM)\" -> Code : 3\n"
            "Variantes : \"consoil (HM)|conseil (PG)|consel (FRA)\" -> Code : 1\n"
            "Variantes : \"consel (HPFGRMA)|secors (V)|confort (S)\" -> Code : 6\n"
            "Variantes : \"jaianz (H)|jaienz (An)|gaians (PS)|jaians (VF)|geans (G)|jaiens (A)\" -> Code : 1\n"
            "Variantes : \"pri (HFGASR)|depri (P)\" -> Code : 5\n"
            "Variantes : \"fox (HAnVGP)|faus (F)|fel (S)\" -> Code : 7\n"
            "Variantes : \"enmainne (HSA)|enmaine (PFMod)|enmoine (G)|emmaine (R)|enmeine (M)\" -> Code : 1\n"
            "Variantes : \"quanque (AnPVFGS)|ce que (H)\" -> Code : 4\n"
            "Variantes : \"En sa maison (HAnPFGAS)|Et as plus vils (V)\" -> Code : 8\n"
            "Variantes : \"poons (HAnPVFG)|poommes (S)\" -> Code : 2\n"
            "Variantes : \"je (HPG)|ge (V)|jou (FS)|jo (AR)\" -> Code : 1\n"
            "Variantes : \"come (HAnPVFGSR)|que (A)\" -> Code : 4\n\n"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We append definitions and target execution prompt *at the end* of the context block
        target_instance = (
            f"Voici les variantes à classifier :\n"
            f"Variantes : \"{self.df.iloc[idx]['readings']}\"\n\n"
            "Choisis STRICTEMENT le code numérique correspondant à la catégorie valide :\n"
            "1 : graph (variation graphémique, ex. païs|pays)\n"
            "2 : flex (variation flexionnelles, ex. savomes|savons)\n"
            "3 : morsynt (variation morphosyntaxique, ex. les despose|le deposera)\n"
            "4 : semlex:minor:gramm (mot-outil ou grammème, ex. cele|l’)\n"
            "5 : semlex:minor:constrmorph (morphologie constructionnelle, ex. mervoillent|esmerveillent)\n"
            "6 : semlex:minor:syn (synonymie, ex. trichier|decevoir)\n"
            "7 : semlex:minor:semsim (proximité sémantique, co-hyponimie, ex. sergant|archir)\n"
            "8 : semlex:major (écart sémantico-lexical fort, ex. acorde|discorde)\n\n"
            "Réponds uniquement avec le chiffre. Aucun commentaire.\n"
            "Code : [/INST]"
        )
        return {"prompt": self.prefix + target_instance}


# Initialize Model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
tokenizer.pad_token = tokenizer.eos_token

# Device execution with 16-bit floats to conserve memory
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
model.config.pad_token_id = tokenizer.eos_token_id

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, clean_up_tokenization_spaces=False)

# Explicit target mappings for numeric tokens 1 through 8
DIGIT_TARGETS = ["1", "2", "3", "4", "5", "6", "7", "8"]
allowed_token_ids = [tokenizer.encode(d, add_special_tokens=False)[-1] for d in DIGIT_TARGETS]
logits_processors = LogitsProcessorList([RestrictedDigitsProcessor(allowed_token_ids)])

# Data Pipeline Setup
df_variants = pd.read_csv("Yvain_reviewed_2026_ids.tsv", sep="\t")
dataset = FewShotVariantDataset(df_variants)

y_true = [normalize_true_label(lbl) for lbl in df_variants["type"]]
y_pred = []

# Map digits safely back to category names for report building
INT_TO_LABEL = {
    "1": "graph",
    "2": "flex",
    "3": "morsynt",
    "4": "semlex:minor:gramm",
    "5": "semlex:minor:constrmorph",
    "6": "semlex:minor:syn",
    "7": "semlex:minor:semsim",
    "8": "semlex:major"
}

print("Running memory-optimized inference loop...")

# Dropping batch size down to 8 balances the huge contextual footprint of our long prompt
outputs = generator(
    KeyDataset(dataset, "prompt"),
    batch_size=8,
    return_full_text=False,
    generation_config=GenerationConfig(
        max_new_tokens=1,  # Set back to 1! The processor guarantees it's a clean digit selection.
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    ),
    logits_processor=logits_processors
)

for out in tqdm(outputs, total=len(df_variants)):
    pred_char = out[0]["generated_text"].strip()
    digit = ''.join(filter(str.isdigit, pred_char))
    y_pred.append(INT_TO_LABEL.get(digit, "other"))

print("\n=== FINAL CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, zero_division=0))