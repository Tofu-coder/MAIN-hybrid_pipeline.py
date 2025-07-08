# MAIN-hybrid_pipeline.py
This is the actual script that follows the REAMDME.md

import os, subprocess, time
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForCausalLM

# === CONFIGURATION ===
DATA_DIR = "data/raw"
PROMPT_DIR = "prompts"
RESULTS_DIR = "results"
SKIP_DOWNLOAD = True  # Skip Entrez + GEO downloading

BIOMODEL = "microsoft/BioGPT-Large"
OLLAMA_MODEL = "llama3"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[INFO] Loading BioGPT model...")
tokenizer = AutoTokenizer.from_pretrained(BIOMODEL)
biogpt = AutoModelForCausalLM.from_pretrained(BIOMODEL)

TF_MODEL = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
TF_MODEL.compile(optimizer='adam', loss='binary_crossentropy')

def run_llama(prompt: str) -> str:
    print("[DEBUG] Starting run_llama()")
    for attempt in range(2):  # retry once
        proc = subprocess.Popen(
            ["ollama", "run", OLLAMA_MODEL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            out, err = proc.communicate(prompt, timeout=90)
            if err.strip():
                print("[LLAMA ERROR]", err.strip())
            return out.strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            print("[ERROR] Ollama call timed out.")
    return ""

def run_biogpt(text: str) -> str:
    try:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outs = biogpt.generate(**inputs, max_new_tokens=128)
            return tokenizer.decode(outs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] run_biogpt exception: {e}")
        return ""

def run_tf(vec):
    try:
        return float(TF_MODEL(tf.convert_to_tensor([vec], tf.float32)).numpy()[0][0])
    except Exception as e:
        print(f"[ERROR] run_tf exception: {e}")
        return 0.0

def clean(name):
    return name.replace(".", "_").replace(" ", "_")

def main():
    if not SKIP_DOWNLOAD:
        print("[PIPELINE] Downloading GEO datasets")
        from Bio import Entrez
        Entrez.email = "you@example.com"  # Update with your email

        def download_geo(geo_id):
            from Bio import Entrez
            handle = Entrez.esearch(db="gds", term=f"{geo_id}[Accession] AND suppFile[Filter]")
            record = Entrez.read(handle)
            handle.close()
            # You can add download logic back if needed

        GEO_IDS = ["GSE264537", "GSE26246", "GSE43312", "GSE48054", "GSE275235"]
        for gid in GEO_IDS:
            download_geo(gid)

    print("\n[PIPELINE] Starting hybrid pipeline")

    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith((".tsv", ".csv"))]
    prompts = [f for f in os.listdir(PROMPT_DIR) if f.endswith(".txt")]

    pairs = []
    for df in data_files:
        for pr in prompts:
            if pr.lower().replace("_prompt.txt", "") in df.lower():
                pairs.append((df, pr))

    print(f"[INFO] Total pairs: {len(pairs)}")
    if not pairs:
        print("[WARN] No data-prompt pairs matched.")
        return

    for idx, (df, pr) in enumerate(pairs, 1):
        print(f"\n[PROCESSING {idx}/{len(pairs)}] Data file: {df}, Prompt file: {pr}")
        df_path = os.path.join(DATA_DIR, df)
        pr_path = os.path.join(PROMPT_DIR, pr)

        try:
            data = open(df_path).read()[:512]
        except Exception as e:
            print(f"[ERROR] reading data file {df}: {e}")
            continue

        try:
            template = open(pr_path).read()
        except Exception as e:
            print(f"[ERROR] reading prompt {pr}: {e}")
            continue

        if "{data}" not in template:
            print(f"[WARN] Template missing {{data}} placeholder: {pr}")
            continue

        prompt = template.replace("{data}", data)
        llama_output = run_llama(prompt)
        print(f"[LLaMA OUTPUT]: {llama_output[:60]}...")

        biogpt_output = run_biogpt(llama_output)
        print(f"[BioGPT OUTPUT]: {biogpt_output[:60]}...")

        score = run_tf([0.2, 0.4, 0.6])

        result = (
            f"=== {pr} x {df} ===\n\n"
            f"[LLaMA]\n{llama_output}\n\n"
            f"[BioGPT]\n{biogpt_output}\n\n"
            f"[TF SCORE] {score:.4f}"
        )
        out_file = os.path.join(RESULTS_DIR, f"{clean(df)}__{clean(pr)}_hybrid.txt")
        with open(out_file, "w") as f:
            f.write(result)
        print("[DONE]", out_file)

    print("\n[PIPELINE] Completed.")

if __name__ == "__main__":
    main()
