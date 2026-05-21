"""
evaluator.py — LLM-as-a-Judge evaluator for debate vs. single-model outputs.

Usage:
    python evaluator.py

The script reads all .txt files from the output folders, groups them by topic,
and asks the local LLM to score each answer on four criteria.
Results are printed to console and saved to: evaluation_results/results_<timestamp>.txt

Scoring criteria (each 1–10):
    1. Comprehensiveness  — number of distinct arguments covered
    2. Balance            — whether multiple perspectives are represented
    3. Concrete examples  — use of real named examples (people, companies, studies)
    4. Overall quality    — overall usefulness as an answer to the topic

Also computes two automatic metrics without LLM:
    - Word count
    - Named entity count (rough: capitalised multi-word phrases)
"""

import os
import re
import json
import glob
import datetime
import requests

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL   = "http://localhost:1234/v1"
MODEL_ID   = "qwen2.5-14b-instruct-1m"   # change to any model you have loaded
TEMPERATURE = 0.0                          # 0 for deterministic scoring

# Folders to scan for output files
OUTPUT_FOLDERS = [
    "debatu_isvestis",
    "vieno_modelio_isvestis",
    "vieno_modelio_isvestis_prompted",
    "deep_thinking_isvestis",
    "deep_thinking_isvestis_prompted",
]

RESULTS_DIR = "evaluation_results"

# ── Helpers ──────────────────────────────────────────────────────────────────

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
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def word_count(text):
    return len(text.split())


def named_entity_count(text):
    """
    Rough heuristic: count capitalised phrases (2+ words) that look like
    proper nouns — e.g. 'IBM Watson', 'Google DeepMind', 'Paris Agreement'.
    Not a real NER; just a cheap proxy that requires no extra libraries.
    """
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    matches = re.findall(pattern, text)
    return len(set(matches))


def read_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_topic_from_content(content):
    """Try to read the topic line written by our programs."""
    for line in content.splitlines():
        if line.startswith("Topic:"):
            return line.replace("Topic:", "").strip().strip("'\"")
    return "Unknown topic"


def extract_answer_from_content(content):
    """Return everything after the === separator (the actual model answer)."""
    sep = "=" * 10
    parts = content.split(sep, 1)
    if len(parts) == 2:
        return parts[1].strip()
    return content.strip()


def score_with_llm(topic, answer, source_label):
    system_prompt = (
        "You are an impartial evaluator assessing the quality of an answer to a debate topic. "
        "Score the answer on exactly four criteria, each on a scale from 1 to 10. "
        "Return ONLY a valid JSON object with no extra text, explanation, or markdown. "
        "The JSON must have exactly these four keys: "
        "  comprehensiveness, balance, concrete_examples, overall_quality. "
        "Definitions:\n"
        "  comprehensiveness  (1-10): How many distinct arguments or aspects are covered? "
        "1 = single vague statement, 10 = thorough coverage of the topic.\n"
        "  balance            (1-10): Does the answer represent multiple perspectives fairly? "
        "1 = one-sided, 10 = all major sides represented with equal depth.\n"
        "  concrete_examples  (1-10): Are real named examples used (companies, studies, events, people)? "
        "1 = no examples at all, 10 = multiple specific named real-world examples.\n"
        "  overall_quality    (1-10): Overall usefulness and quality as an answer to this topic. "
        "1 = unhelpful, 10 = excellent.\n"
        "Example valid output: {\"comprehensiveness\": 7, \"balance\": 8, \"concrete_examples\": 4, \"overall_quality\": 7}"
    )

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Answer (from: {source_label}):\n{answer[:3000]}"  # cap to avoid context overflow
    )

    raw = call_model(system_prompt, user_prompt)
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        scores = json.loads(clean)
        # Validate keys
        required = {"comprehensiveness", "balance", "concrete_examples", "overall_quality"}
        if not required.issubset(scores.keys()):
            raise ValueError(f"Missing keys in JSON: {scores}")
        return scores
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [WARNING] Could not parse scores for '{source_label}': {e}")
        print(f"  Raw output was: {raw[:200]}")
        return None


def folder_label(folder):
    labels = {
        "debatu_isvestis":                    "Debate system",
        "vieno_modelio_isvestis":             "Single model",
        "vieno_modelio_isvestis_prompted":    "Single model (prompted)",
        "deep_thinking_isvestis":             "Deep thinking model",
        "deep_thinking_isvestis_prompted":    "Deep thinking model (prompted)",
    }
    return labels.get(folder, folder)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.txt")

    all_lines = []

    def log(text=""):
        print(text)
        all_lines.append(str(text))

    log("=" * 70)
    log("LLM-as-a-Judge Evaluator")
    log(f"Evaluator model: {MODEL_ID}")
    log(f"Timestamp: {timestamp}")
    log("=" * 70)

    # Collect all files grouped by topic
    # Structure: { topic_str: [ {folder, label, path, content, answer} ] }
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
            topic = extract_topic_from_content(content)
            answer = extract_answer_from_content(content)

            if topic not in topic_groups:
                topic_groups[topic] = []

            topic_groups[topic].append({
                "folder": folder,
                "label":  folder_label(folder),
                "path":   path,
                "content": content,
                "answer": answer,
            })

    if not topic_groups:
        log("\nNo output files found. Run LLMDebatai.py and runSingleModel.py first.")
        return

    # Score each entry
    grand_results = {}

    for topic, entries in topic_groups.items():
        log(f"\n{'=' * 70}")
        log(f"TOPIC: {topic}")
        log(f"{'=' * 70}")

        topic_results = []

        for entry in entries:
            label = entry["label"]
            answer = entry["answer"]
            path = entry["path"]

            log(f"\n  Source : {label}")
            log(f"  File   : {os.path.basename(path)}")

            # Automatic metrics
            wc = word_count(answer)
            ne = named_entity_count(answer)
            log(f"  Words  : {wc}")
            log(f"  Named entities (approx): {ne}")

            # LLM scoring
            log(f"  Scoring with LLM...")
            scores = score_with_llm(topic, answer, label)

            if scores:
                log(f"  Comprehensiveness : {scores['comprehensiveness']}/10")
                log(f"  Balance           : {scores['balance']}/10")
                log(f"  Concrete examples : {scores['concrete_examples']}/10")
                log(f"  Overall quality   : {scores['overall_quality']}/10")
                avg = sum(scores.values()) / len(scores)
                log(f"  Average score     : {avg:.1f}/10")
            else:
                log("  [Scoring failed for this entry]")

            topic_results.append({
                "label":   label,
                "file":    os.path.basename(path),
                "words":   wc,
                "named_entities": ne,
                "scores":  scores,
            })

        grand_results[topic] = topic_results

        # Summary table for this topic
        log(f"\n  --- Summary table for this topic ---")
        header = f"  {'Source':<35} {'Words':>6} {'NE':>4} {'Comp':>5} {'Bal':>5} {'Ex':>5} {'Qual':>5} {'Avg':>5}"
        log(header)
        log("  " + "-" * (len(header) - 2))

        for r in topic_results:
            s = r["scores"]
            if s:
                avg = sum(s.values()) / len(s)
                row = (
                    f"  {r['label']:<35} "
                    f"{r['words']:>6} "
                    f"{r['named_entities']:>4} "
                    f"{s['comprehensiveness']:>5} "
                    f"{s['balance']:>5} "
                    f"{s['concrete_examples']:>5} "
                    f"{s['overall_quality']:>5} "
                    f"{avg:>5.1f}"
                )
            else:
                row = f"  {r['label']:<35} {r['words']:>6} {r['named_entities']:>4}   N/A"
            log(row)

    # Overall summary across all topics
    log(f"\n{'=' * 70}")
    log("OVERALL AVERAGE SCORES (across all topics)")
    log("=" * 70)

    # Aggregate by label
    label_totals = {}
    for topic, entries in grand_results.items():
        for r in entries:
            lbl = r["label"]
            if lbl not in label_totals:
                label_totals[lbl] = {"comp": [], "bal": [], "ex": [], "qual": [], "words": [], "ne": []}
            label_totals[lbl]["words"].append(r["words"])
            label_totals[lbl]["ne"].append(r["named_entities"])
            if r["scores"]:
                label_totals[lbl]["comp"].append(r["scores"]["comprehensiveness"])
                label_totals[lbl]["bal"].append(r["scores"]["balance"])
                label_totals[lbl]["ex"].append(r["scores"]["concrete_examples"])
                label_totals[lbl]["qual"].append(r["scores"]["overall_quality"])

    def avg_or_na(lst):
        return f"{sum(lst)/len(lst):.1f}" if lst else "N/A"

    log(f"\n  {'Source':<35} {'Words':>6} {'NE':>4} {'Comp':>5} {'Bal':>5} {'Ex':>5} {'Qual':>5}")
    log("  " + "-" * 66)
    for lbl, d in label_totals.items():
        log(
            f"  {lbl:<35} "
            f"{avg_or_na(d['words']):>6} "
            f"{avg_or_na(d['ne']):>4} "
            f"{avg_or_na(d['comp']):>5} "
            f"{avg_or_na(d['bal']):>5} "
            f"{avg_or_na(d['ex']):>5} "
            f"{avg_or_na(d['qual']):>5}"
        )

    log()
    log("Comp = Comprehensiveness, Bal = Balance, Ex = Concrete examples, Qual = Overall quality")

    # Save results
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

    print(f"\n[Results saved to: {results_file}]")


if __name__ == "__main__":
    main()