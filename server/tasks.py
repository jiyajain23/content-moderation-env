"""
Task definitions for the Content Moderation Environment.

Three tasks with increasing difficulty:
  1. EASY   – obvious spam
  2. MEDIUM – clearly abusive language
  3. HARD   – ambiguous, context-dependent content
"""

from typing import List, Dict, Any

PLATFORM_RULES = [
    "No unsolicited commercial advertisements or spam.",
    "No hate speech, slurs, or targeted harassment of individuals.",
    "No threats of violence or incitement to harm.",
    "Repeated policy violations result in escalating enforcement (warn → remove → ban).",
    "Context matters: satire and news reporting may discuss sensitive topics without violating rules.",
    "User history of violations is a key signal when borderline content is detected.",
]

TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "task_easy_001",
        "difficulty": "easy",
        "post_id": "post_001",
        "content": (
            "MAKE $5,000 A DAY FROM HOME! Click here NOW -> http://totally-legit-money.biz "
            "LIMITED SPOTS! Send $50 to unlock your FREE starter kit. "
            "DM me for details. #WorkFromHome #EasyMoney #GetRichQuick"
        ),
        "user_history": [
            "Earn passive income! Visit my link in bio.",
            "Join my exclusive money-making group – only 10 spots left!",
        ],
        "flags": [
            "Reported as spam by 12 users",
            "Link flagged by automated URL scanner as phishing",
        ],
        "platform_rules": PLATFORM_RULES,
        "ground_truth_label": "spam",
        "ground_truth_decision": "remove",
        "description": (
            "A textbook spam post with promotional language, suspicious URLs, "
            "and a consistent spam history. Classification and removal are unambiguous."
        ),
    },
    {
        "task_id": "task_medium_001",
        "difficulty": "medium",
        "post_id": "post_002",
        "content": (
            "@jessica_m you are absolutely worthless and everyone hates you. "
            "Do the world a favour and disappear. Nobody wants you here, you stupid idiot."
        ),
        "user_history": [
            "Normal post: shared a recipe last week.",
            "Argued with another user two days ago – no policy violation.",
        ],
        "flags": [
            "Reported for harassment by the targeted user",
            "Reported by 3 additional community members",
        ],
        "platform_rules": PLATFORM_RULES,
        "ground_truth_label": "abusive",
        "ground_truth_decision": "remove",
        "description": (
            "Direct targeted harassment with personal insults. "
            "No satire or news context. Clear abuse requiring removal."
        ),
    },
    {
        "task_id": "task_hard_001",
        "difficulty": "hard",
        "post_id": "post_003",
        "content": (
            "I can't believe people still defend him. "
            "He's a cancer on this community and needs to be cut out before he destroys everything. "
            "We all know what he did – don't let him get away with it again."
        ),
        "user_history": [
            "Posted about local politics regularly over 2 years – no violations.",
            "Previously warned once for heated language in a political debate (6 months ago).",
            "Active moderator in two community groups.",
        ],
        "flags": [
            "One report: 'This sounds like a threat'",
            "Original poster clarified in comments: referring to a local politician's policy record",
        ],
        "platform_rules": PLATFORM_RULES,
        "ground_truth_label": "safe",
        "ground_truth_decision": "allow",
        "description": (
            "Figurative political criticism that sounds alarming in isolation "
            "but is clearly metaphorical given user history and context clarification. "
            "Agent must use context to avoid over-moderation."
        ),
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"Task '{task_id}' not found. Available: {[t['task_id'] for t in TASKS]}")


def get_all_task_ids() -> List[str]:
    return [t["task_id"] for t in TASKS]
