import json
import requests

# The local model server address
BASE_URL = "http://localhost:1234/v1"

# The single model used for every role
MODEL_ID = "qwen2.5-14b-instruct-1m"

# How many speakers each team gets.
SPEAKERS_PER_TEAM = 2

# Temperature for debate speeches.
TEMPERATURE = 0.7

# Temperature for the team-generation step.
ANALYSIS_TEMPERATURE = 0.3

TOPIC = "Is 0.999... equal to 1?"


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


# Step 1: Ask the AI how many teams are needed and what their stances are

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
        "Return the JSON array of teams now."
    )

    raw = call_model(system_prompt, user_prompt, temperature=ANALYSIS_TEMPERATURE)

    # Strip accidental markdown fences if the model adds them despite instructions
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
        "Use evidence and logic. Be persuasive but concise. "
        "You must respond in English only, regardless of your default language."
    )

    if speaker_number == 1:
        prompt += (
            "You are the opening speaker. "
            "Introduce your team's core position and lay out your main arguments. "
        )

    if speaker_number == total_speakers:
        prompt += (
            "You are the closing speaker. "
            "After your arguments, summarise your team's overall position. "
        )

    return prompt


def build_final_prompt(topic, teams):
    team_list = ", ".join(t["name"] for t in teams)

    return (
        f"You are an analyst reviewing a multi-team debate on: '{topic}'. "
        f"The teams were: {team_list}. "
        "Your job is to summerize the debate. "
        "Give an answer for the topic, use arguments from both sides. "
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
    print(f"\nTopic: '{TOPIC}'")
    print("Asking the model to identify debate sides...\n")

    teams = generate_teams(TOPIC)

    print(f"Teams generated ({len(teams)} sides):")
    for t in teams:
        print(f"  {t['name']}: {t['stance']}")

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
        print(f"\n{label}:\n{reply}\n")
        debate_log += f"\n\n{label}:\n{reply}"

    print("FINAL OUTPUT")

    user_prompt = (
        f"The debate topic is: '{TOPIC}'.\n\n"
        f"Full debate log:\n\n{debate_log}\n\n"
        "Provide your structured answer to the topic."
    )
    finalOutput = call_model(build_final_prompt(TOPIC, teams), user_prompt)

    print(f"\n{finalOutput}\n")

    return teams, debate_log, finalOutput


# Entry point

if __name__ == "__main__":
    run_debate()