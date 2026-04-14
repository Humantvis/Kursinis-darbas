import requests

TEAM_SPEAKERS = 2
BASE_URL = "http://localhost:1234/v1"

team1_speakers = ["t1s1", "t1s2"]
team2_speakers = ["t2s1", "t2s2"]
judges = ["judge"]

if len(team1_speakers) != TEAM_SPEAKERS or len(team2_speakers) != TEAM_SPEAKERS:
    raise ValueError("Number of speakers for each team must match TEAM_SPEAKERS")

debate_log = ""

topic = "AI is beneficial for humanity"


def run_model(model_id, messages):
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
    }

    resp = requests.post(url, json=payload)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        print("Status code:", resp.status_code)
        print("Response text:", resp.text)
        raise

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def give_speaker_role(team, speaker, topic):
    stance = "for" if team == 1 else "against"

    prompt = (
        f"You are a debater arguing {stance} the proposition that '{topic}'. "
        f"Do not add 'Team {team} Speaker {speaker}:' to the start of your response; just give your speech. "
        "You may reply to the other team and rebut their arguments. "
        "You can use anything that was said previously in the conversation. "
    )

    if speaker == 1:
        prompt += (
            f"You are the first speaker for your team of {TEAM_SPEAKERS} speakers. "
            "Make an opening statement/introduction for your team. "
        )

    if speaker == TEAM_SPEAKERS:
        prompt += (
            "You are the last speaker for your team. "
            "After your speech, summarize your team's position."
        )

    return prompt


def give_judge_role(topic):
    prompt = (
        f"You are a judge for a debate on the topic '{topic}'. "
        "The debate has concluded. Please provide a decision on which team won and a brief explanation for your decision."
    )
    return prompt


print(f"Starting the debate on the topic: '{topic}'\n")


for i in range(TEAM_SPEAKERS):
    team1_messages = [
        {"role": "system", "content": give_speaker_role(1, i + 1, topic)},
        {
            "role": "user",
            "content": (
                f"The debate topic is: '{topic}'.\n\n"
                f"Here is the debate so far:\n\n"
                f"{debate_log}\n\n"
                "Give your next speech for Team 1."
            ),
        },
    ]

    team1_reply = run_model(team1_speakers[i], team1_messages)
    print(f"\n\nTeam 1 Speaker {i + 1}: {team1_reply}")
    debate_log += f"\n\nTeam 1 Speaker {i + 1}: {team1_reply}"

    team2_messages = [
        {"role": "system", "content": give_speaker_role(2, i + 1, topic)},
        {
            "role": "user",
            "content": (
                f"The debate topic is: '{topic}'.\n\n"
                f"Here is the debate so far:\n\n"
                f"{debate_log}\n\n"
                "Now give your next speech for Team 2."
            ),
        },
    ]

    team2_reply = run_model(team2_speakers[i], team2_messages)
    print(f"\n\nTeam 2 Speaker {i + 1}: {team2_reply}")
    debate_log += f"\n\nTeam 2 Speaker {i + 1}: {team2_reply}"


for judge in judges:
    judge_messages = [
        {"role": "system", "content": give_judge_role(topic)},
        {
            "role": "user",
            "content": (
                f"The debate topic is: '{topic}'.\n\n"
                f"Here is the full debate:\n\n"
                f"{debate_log}\n\n"
                "Please provide your judgment on which team won and why. Start your response with 'Team 1 wins' or 'Team 2 wins' followed by your explanation."
            ),
        },
    ]

    judge_reply = run_model(judge, judge_messages)
    print(f"\n\nJudge's Decision: {judge_reply}")