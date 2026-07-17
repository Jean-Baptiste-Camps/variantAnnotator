from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

# Charger Mistral 7B (ou une version plus légère si GPU limité)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Utiliser FP16 pour économiser la mémoire
    device_map="auto"  # Répartir automatiquement sur le GPU
)

# Créer un pipeline de génération
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cuda"  # Utiliser le GPU
)

#TODO: on laisse tomber le contexte et les "nonsense" readings pour le moment
def classify_variant(readings, types_autorises):
    prompt = f"""
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

    # Générer la réponse
    outputs = generator(
        prompt,
        max_new_tokens=20,  # Limiter la longueur de la réponse
        temperature=0.0,  # Pour une réponse déterministe
        do_sample=False
    )
    predicted_type = outputs[0]["generated_text"].strip().split("\n")[-1].strip()
    return predicted_type

df_variants = pd.read_csv("Yvain_reviewed_2026_ids.tsv", sep="\t")

# Exemple d'utilisation
sample_variant = df_variants.iloc[0]
predicted_type = classify_variant(
#    sample_variant["context"],
    sample_variant["readings"]#,
#    types_autorises
)
print(f"Type prédit : {predicted_type}")