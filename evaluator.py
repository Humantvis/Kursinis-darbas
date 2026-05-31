import os
import re
import json
import glob
import datetime
import requests


BASE_URL    = "http://localhost:1234/v1"
MODEL_ID    = "qwen2.5-14b-instruct-1m"
TEMPERATURE = 0.0

OUTPUT_FOLDERS = [
    "debatu_isvestis",
    "vieno_modelio_isvestis",
    "vieno_modelio_isvestis_prompted",
    "deep_thinking_isvestis",
    "deep_thinking_isvestis_prompted",
]

RESULTS_DIR = "evaluation_results"

GRADING_INSTRUCTIONS_FILE = "grading_instructions.txt"

# Weights must sum to 1.0.
WEIGHTS = {
    "argument_coverage":       0.35,
    "argument_depth":          0.20,
    "balanced_representation": 0.20,
    "semantic_relevance":      0.15,
    "factual_consistency":     0.10,
}


def call_model(system_prompt, user_prompt):
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
    }
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def word_count(text):
    return len(text.split())

def named_entity_count(text):
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    return len(set(re.findall(pattern, text)))


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def extract_topic(content):
    for line in content.splitlines():
        if line.startswith("Topic:"):
            return line.replace("Topic:", "").strip().strip("'\"")
    return "Unknown topic"

def extract_answer(content):
    sep = "=" * 10
    parts = content.split(sep, 1)
    return parts[1].strip() if len(parts) == 2 else content.strip()

def folder_label(folder):
    return {
        "debatu_isvestis":                 "Debate system",
        "vieno_modelio_isvestis":          "Single model",
        "vieno_modelio_isvestis_prompted": "Single model (prompted)",
        "deep_thinking_isvestis":          "Deep thinking model",
        "deep_thinking_isvestis_prompted": "Deep thinking model (prompted)",
    }.get(folder, folder)


def generate_reference_arguments(topic):
    system_prompt = (
        "You are an expert debate analyst. "
        "Given a debate topic, produce a comprehensive reference argument set covering "
        "the strongest arguments FOR the proposition, AGAINST it, and any important "
        "nuanced positions. Aim for 10-14 arguments total. "
        "Return ONLY a JSON array of strings. No explanation, no markdown, no backticks. "
        "Each string is one argument, concise (one sentence). "
        'Example: ["AI improves medical diagnostics", "AI causes job displacement", ...]'
    )
    user_prompt = f"The debate topic is: '{topic}'. Generate the reference argument set now."

    raw = call_model(system_prompt, user_prompt)
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        args = json.loads(clean)
        if isinstance(args, list) and all(isinstance(a, str) for a in args):
            return args
    except json.JSONDecodeError:
        pass
    print(f"  [WARNING] Could not parse reference arguments. Raw: {raw[:200]}")
    return []


def load_grading_instructions():
    if not os.path.isfile(GRADING_INSTRUCTIONS_FILE):
        raise FileNotFoundError(
            f"Grading instructions file not found: '{GRADING_INSTRUCTIONS_FILE}'\n"
            f"Make sure '{GRADING_INSTRUCTIONS_FILE}' is in the same directory as evaluator.py."
        )
    with open(GRADING_INSTRUCTIONS_FILE, encoding="utf-8") as f:
        return f.read()

def score_with_llm(topic, answer, source_label, reference_args, system_prompt):
    ref_block = "\n".join(f"  - {a}" for a in reference_args) if reference_args else "  (none provided)"

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Reference argument set:\n{ref_block}\n\n"
        f"Response to grade (source: {source_label}):\n{answer[:4000]}"
    )

    raw = call_model(system_prompt, user_prompt)
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    required_keys = {
        "argument_coverage", "argument_depth", "balanced_representation",
        "semantic_relevance", "factual_consistency",
    }

    try:
        scores = json.loads(clean)
        missing = required_keys - scores.keys()
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        return scores
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [WARNING] Could not parse scores for '{source_label}': {e}")
        print(f"  Raw output: {raw[:300]}")
        return None


def composite(scores):
    if scores is None:
        return None

    fc = scores.get("factual_consistency")
    factual_na = (fc == "N/A" or fc is None)

    w = dict(WEIGHTS)
    if factual_na:
        w["argument_coverage"] += w.pop("factual_consistency")
    else:
        w = dict(WEIGHTS)

    total = 0.0
    for key, weight in w.items():
        val = scores.get(key)
        if val is None or val == "N/A":
            continue
        total += float(val) * weight

    return round(total, 1)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.txt")

    lines = []
    def log(text=""):
        print(text)
        lines.append(str(text))

    log("=" * 70)
    log("LLM-as-a-Judge Evaluator")
    log(f"Evaluator model       : {MODEL_ID}")
    log(f"Grading instructions  : {GRADING_INSTRUCTIONS_FILE}")
    log(f"Timestamp             : {timestamp}")
    log("=" * 70)

    system_prompt = load_grading_instructions()
    log(f"\n[OK] Loaded grading instructions ({len(system_prompt)} chars) from '{GRADING_INSTRUCTIONS_FILE}'")

    topic_groups = {}
    for folder in OUTPUT_FOLDERS:
        if not os.path.isdir(folder):
            log(f"\n[SKIP] Folder not found: {folder}")
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not files:
            log(f"\n[SKIP] No .txt files in: {folder}")
            continue
        for path in files:
            content = read_file(path)
            topic = extract_topic(content)
            answer = extract_answer(content)
            topic_groups.setdefault(topic, []).append({
                "folder":  folder,
                "label":   folder_label(folder),
                "path":    path,
                "answer":  answer,
            })

    if not topic_groups:
        log("\nNo output files found. Run LLMDebatai.py and runSingleModel.py first.")
        return

    grand_results = {}

    for topic, entries in topic_groups.items():
        log(f"\n{'=' * 70}")
        log(f"TOPIC: {topic}")
        log(f"{'=' * 70}")

        # Generate reference arguments once per topic
        log("\n  Generating reference argument set for this topic...")
        ref_args = generate_reference_arguments(topic)
        if ref_args:
            log(f"  Reference arguments ({len(ref_args)}):")
            for a in ref_args:
                log(f"    - {a}")
        else:
            log("  [WARNING] No reference arguments generated; Coverage scoring will be limited.")
        log()

        topic_results = []

        for entry in entries:
            label  = entry["label"]
            answer = entry["answer"]
            path   = entry["path"]

            log(f"  Source : {label}")
            log(f"  File   : {os.path.basename(path)}")

            wc = word_count(answer)
            ne = named_entity_count(answer)
            log(f"  Words  : {wc}  |  Named entities (approx): {ne}")
            log("  Scoring with LLM...")

            scores = score_with_llm(topic, answer, label, ref_args, system_prompt)

            if scores:
                fc_display = scores['factual_consistency']
                if fc_display != "N/A":
                    fc_display = f"{float(fc_display):.1f}/10"
                else:
                    fc_display = "N/A"

                log(f"  Argument Coverage         : {float(scores['argument_coverage']):.1f}/10")
                log(f"  Argument Depth            : {float(scores['argument_depth']):.1f}/10")
                log(f"  Balanced Representation   : {float(scores['balanced_representation']):.1f}/10")
                log(f"    PRO units: {scores.get('pro_count', '?')}  |  CON units: {scores.get('con_count', '?')}")
                log(f"  Semantic Relevance        : {float(scores['semantic_relevance']):.1f}/10")
                log(f"    {scores.get('relevance_justification', '')}")
                log(f"  Factual Consistency       : {fc_display}")

                cd = scores.get("coverage_detail", {})
                if cd:
                    log(f"  Coverage — covered     : {cd.get('covered', [])}")
                    log(f"  Coverage — not covered : {cd.get('not_covered', [])}")

                comp = composite(scores)
                log(f"  ── Composite score        : {comp}/10")
            else:
                comp = None
                log("  [Scoring failed for this entry]")

            log()
            topic_results.append({
                "label":          label,
                "file":           os.path.basename(path),
                "words":          wc,
                "named_entities": ne,
                "scores":         scores,
                "composite":      comp,
            })

        grand_results[topic] = topic_results

        log(f"  {'─' * 66}")
        log(f"  Summary — {topic[:50]}")
        header = f"  {'Source':<35} {'Wds':>5} {'Cov':>5} {'Dep':>5} {'Bal':>5} {'Rel':>5} {'Fct':>5} {'Cmp':>5}"
        log(header)
        log("  " + "─" * (len(header) - 2))
        for r in topic_results:
            s = r["scores"]
            if s:
                fc = s['factual_consistency']
                fc_str = "N/A" if fc == "N/A" else f"{float(fc):.1f}"
                row = (
                    f"  {r['label']:<35}"
                    f"{r['words']:>5} "
                    f"{float(s['argument_coverage']):>5.1f} "
                    f"{float(s['argument_depth']):>5.1f} "
                    f"{float(s['balanced_representation']):>5.1f} "
                    f"{float(s['semantic_relevance']):>5.1f} "
                    f"{fc_str:>5} "
                    f"{r['composite']:>5.1f}"
                )
            else:
                row = f"  {r['label']:<35}{r['words']:>5}  scoring failed"
            log(row)
        log(f"  Cov=Coverage Dep=Depth Bal=Balance Rel=Relevance Fct=Factual Cmp=Composite")

    log(f"\n{'=' * 70}")
    log("OVERALL AVERAGE SCORES (across all topics and runs)")
    log("=" * 70)

    label_totals = {}
    for topic, entries in grand_results.items():
        for r in entries:
            lbl = r["label"]
            d = label_totals.setdefault(lbl, {
                "coverage": [], "depth": [], "balance": [],
                "relevance": [], "factual": [], "composite": [],
                "words": [], "ne": [],
            })
            d["words"].append(r["words"])
            d["ne"].append(r["named_entities"])
            if r["composite"] is not None:
                d["composite"].append(r["composite"])
            s = r["scores"]
            if s:
                d["coverage"].append(float(s["argument_coverage"]))
                d["depth"].append(float(s["argument_depth"]))
                d["balance"].append(float(s["balanced_representation"]))
                d["relevance"].append(float(s["semantic_relevance"]))
                fc = s["factual_consistency"]
                if fc != "N/A" and fc is not None:
                    d["factual"].append(float(fc))

    def avg(lst):
        return f"{sum(lst)/len(lst):.1f}" if lst else " N/A"

    log(f"\n  {'Source':<35} {'Wds':>5} {'Cov':>5} {'Dep':>5} {'Bal':>5} {'Rel':>5} {'Fct':>5} {'Cmp':>5}")
    log("  " + "─" * 68)
    for lbl, d in label_totals.items():
        log(
            f"  {lbl:<35}"
            f"{avg(d['words']):>5} "
            f"{avg(d['coverage']):>5} "
            f"{avg(d['depth']):>5} "
            f"{avg(d['balance']):>5} "
            f"{avg(d['relevance']):>5} "
            f"{avg(d['factual']):>5} "
            f"{avg(d['composite']):>5}"
        )

    log()
    log("Cov=Coverage  Dep=Depth  Bal=Balance  Rel=Relevance  Fct=Factual  Cmp=Composite")
    log("Weights: Coverage 35%  Depth 20%  Balance 20%  Relevance 15%  Factual 10%")
    log("(Factual weight redistributed to Coverage when N/A)")

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[Results saved to: {results_file}]")


if __name__ == "__main__":
    main()