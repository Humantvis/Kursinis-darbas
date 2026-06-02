import json
import os
import datetime
import requests

# The local model server address
BASE_URL = "http://localhost:1234/v1"

# The single model used for every role
MODEL_ID = "qwen2.5-14b-instruct-1m"

# How many speakers each team gets.
SPEAKERS_PER_TEAM = 2

# Temperature for debate speeches.
TEMPERATURE = 0.7

# Temperature for the team-generation step and final analysis.
ANALYSIS_TEMPERATURE = 0.5

TOPIC = "Animals should have the same legal rights as humans"

# Output folder
OUTPUT_DIR = "debatu_isvestis"


def call_model(system_prompt, user_prompt, temperature=None):
    if temperature is None:
        temperature = TEMPERATURE

    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(url, json=payload)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        print("HTTP error:", response.status_code)
        print("Response body:", response.text)
        raise

    return response.json()["choices"][0]["message"]["content"]


# Step 1: How many teams, what are their stances

def generate_teams(topic):
    system_prompt = (
        "You are a debate organiser. "
        "Your job is to analyse a debate topic and think of all distinct perspectives that there are on the topic. "
        "A topic does not always have two sides(for and against) - some topics have three or more different positions. "
        "But, do not invent sides that do not have a meaning to exist. "
        "Return ONLY a JSON array and nothing else. "
        "No explanation, no markdown, no backticks. "
        "Each item must have exactly two keys: "
        "  'name'   — e.g. 'Team 1', 'Team 2', etc. "
        "  'stance' — a short phrase describing their position, used in the sentence: "
        "             'I am arguing [stance] the proposition'. "
        "Example output for a two-sided topic: "
        '[{"name":"Team 1","stance":"strongly in favour of"},'
        '{"name":"Team 2","stance":"strongly against"}]'
    )

    user_prompt = (
        f"The debate topic is: '{topic}'. "
        "How many distinct sides does this topic have? "
        "Return the JSON array of teams."
    )

    raw = call_model(system_prompt, user_prompt, temperature=ANALYSIS_TEMPERATURE)

    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        teams = json.loads(clean)
    except json.JSONDecodeError:
        print("Could not parse team JSON. Raw model output was:")
        print(raw)
        raise

    for i, team in enumerate(teams):
        if "name" not in team or "stance" not in team:
            raise ValueError(f"Team at index {i} is missing 'name' or 'stance': {team}")

    return teams


# Step 2: Build prompts for each role

def build_speaker_prompt(team, speaker_number, total_speakers, topic):
    prompt = (
        f"You are a debater representing {team['name']}. "
        f"You are arguing {team['stance']} the proposition: '{topic}'. "
        "Speak only for yourself — do not start your reply with your team name. "
        "You may rebut arguments made by other teams. "
        "Support every argument with specific evidence, statistics, named studies, "
        "or concrete real-world examples where possible — do not make unsupported claims. "
        "Cover as many distinct arguments for your position as possible. "
        "You must respond in English only, regardless of your default language."
    )

    if speaker_number == 1:
        prompt += (
            "You are the opening speaker. "
            "Introduce your team's core position and lay out your main arguments. "
        )

    if speaker_number > 1:
        prompt += (
            "You MUST introduce at least two new arguments that have not yet been raised "
            "by anyone in this debate — including your own teammates. "
            "Do not simply rebut or repeat what has already been said. "
            "Expand the scope of the discussion with fresh perspectives. "
        )

    if speaker_number == total_speakers:
        prompt += (
            "You are the closing speaker. "
            "After your new arguments, briefly summarise your team's overall position. "
        )

    return prompt


def build_final_prompt(topic, teams):
    team_list = ", ".join(t["name"] for t in teams)

    return (
        f"You are an expert analyst reviewing a multi-team debate on: '{topic}'. "
        f"The teams were: {team_list}. "
        "Your job is to write a comprehensive, well-structured answer to the topic question. "
        "The debate transcript is provided only as reference material — your answer must go beyond it. "
        "You MUST independently identify and include ALL major arguments on ALL sides of this topic, "
        "whether or not the debaters raised them. Do not limit yourself to what was said in the debate. "
        "For each argument, provide specific evidence, concrete examples, or statistics where possible "
        "— do not merely list claims without support. "
        "Structure your answer with clear sections covering each distinct perspective. "
        "Do not pick a side — represent all perspectives fairly and proportionally. "
        "Your answer must be thorough enough that a reader would not need to read the debate "
        "transcript to understand the full picture of this topic. "
        "You must respond in English only, regardless of your default language."
    )


# Step 3: Build the speaking schedule

def build_speaking_schedule(teams, speakers_per_team):
    schedule = []

    for i in range(speakers_per_team):
        for team in teams:
            schedule.append((team, i + 1))

    return schedule


# Step 4: Run the debate

def run_debate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_chars = []
    for c in TOPIC:
        if c.isalnum() or c in " _-":
            safe_chars.append(c)
        else:
            safe_chars.append("_")
    safe_topic = "".join(safe_chars)[:60].strip()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"{safe_topic}__{timestamp}.txt")

    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log(f"Topic: {TOPIC}")
    log(f"Model: {MODEL_ID}")
    log(f"Speakers per team: {SPEAKERS_PER_TEAM}")
    log(f"Temperature: {TEMPERATURE}  |  Analysis temperature: {ANALYSIS_TEMPERATURE}")
    log()

    teams = generate_teams(TOPIC)

    log(f"Teams generated ({len(teams)} sides):")
    for t in teams:
        log(f"  {t['name']}: {t['stance']}")
    log()

    debate_log = ""
    schedule = build_speaking_schedule(teams, SPEAKERS_PER_TEAM)

    for (team, speaker_number) in schedule:

        system_prompt = build_speaker_prompt(team, speaker_number, SPEAKERS_PER_TEAM, TOPIC)

        user_prompt = (
            f"The debate topic is: '{TOPIC}'.\n\n"
            f"Here is the debate so far:\n\n"
            f"{debate_log if debate_log else '(No speeches yet — you are first.)'}\n\n"
            f"Now give your speech as {team['name']}, speaker {speaker_number}."
        )

        reply = call_model(system_prompt, user_prompt)

        label = f"{team['name']} — Speaker {speaker_number}"
        log(f"\n{label}:\n{reply}\n")
        debate_log += f"\n\n{label}:\n{reply}"

    log("FINAL OUTPUT")
    log("=" * 70)

    analysis_user = (
        f"The debate topic is: '{TOPIC}'.\n\n"
        f"Below is the full debate transcript for reference:\n\n"
        f"<debate_transcript>\n{debate_log}\n</debate_transcript>\n\n"
        "Now write your comprehensive independent analysis of the topic. "
        "Remember: the transcript above is reference material only. "
        "Your answer must stand alone as a complete and balanced treatment of the topic, "
        "covering all major perspectives and arguments regardless of what the debaters said."
    )
    finalOutput = call_model(build_final_prompt(TOPIC, teams), analysis_user,
                             temperature=ANALYSIS_TEMPERATURE)

    log(f"\n{finalOutput}\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[Output saved to: {filename}]")

    return teams, debate_log, finalOutput



if __name__ == "__main__":
    run_debate()