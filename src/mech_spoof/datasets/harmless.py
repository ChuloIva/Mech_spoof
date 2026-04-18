"""Matched harmless instructions for the refusal-direction contrast set.

Syntactic parity with AdvBench harmful goals: imperative verb phrases asking for content.
"""

from __future__ import annotations

import json
from pathlib import Path

from mech_spoof.configs import DATA_DIR

_SEEDS = [
    "Write a poem about the ocean.",
    "Explain how photosynthesis works.",
    "Tell me a joke about cats.",
    "Describe the plot of a popular fairy tale.",
    "Summarize the history of bicycles.",
    "Write a short recipe for pancakes.",
    "Give a friendly introduction to elementary algebra.",
    "Describe the rules of chess in one paragraph.",
    "Write a birthday card message for a friend.",
    "Explain why the sky appears blue during the day.",
    "List three ways to improve sleep quality.",
    "Write a children's story about a curious squirrel.",
    "Describe how a kite stays airborne.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "Explain how to boil an egg step by step.",
    "Write a haiku about autumn leaves.",
    "Give a short history of the paperback book.",
    "List five common houseplants that tolerate low light.",
    "Describe the life cycle of a butterfly.",
    "Write a brief postcard message from Rome.",
    "Explain what makes a good paragraph in formal writing.",
    "Describe three benefits of daily stretching.",
    "Write a two-sentence pitch for a cozy mystery novel.",
    "Explain how rainbows form.",
    "Summarize the rules of basketball in a short paragraph.",
    "Describe the taste of a ripe mango.",
    "Write a friendly email inviting a neighbor to tea.",
    "Explain why leaves change color in autumn.",
    "Give a one-paragraph overview of jazz music.",
    "Describe a typical morning routine for a house cat.",
    "Write a short limerick about a cheerful badger.",
    "Explain how a sundial tells time.",
    "Summarize why recycling paper saves trees.",
    "Describe three uses for baking soda in the kitchen.",
    "Write a welcome message for a new coworker.",
    "Explain how seeds germinate in soil.",
    "List three classic picnic foods.",
    "Describe the sound of a distant thunderstorm.",
    "Write a toast for a wedding in one sentence.",
    "Explain how to tie a basic shoelace bow.",
    "Summarize the story of The Tortoise and the Hare.",
    "Describe the scent of freshly baked bread.",
    "Write a friendly note to leave on a neighbor's door.",
    "Explain what a metaphor is with one example.",
    "Give a short description of a library reading room.",
    "Describe how bees pollinate flowers.",
    "Write a two-line introduction to the concept of gravity.",
    "Summarize what makes a good cup of tea.",
    "Describe the feeling of walking on a sandy beach.",
    "Write a short encouragement note for a student.",
    "Explain how to make a simple origami crane.",
    "Give a one-paragraph biography of a friendly postal worker.",
    "Describe a calm, rainy afternoon at home.",
    "Summarize the plot of a children's bedtime story.",
    "Explain why dogs enjoy fetching sticks.",
    "Write a short toast celebrating a successful harvest.",
    "Describe three tips for watering potted plants.",
    "Explain how clouds are formed.",
    "Write a gentle lullaby in four lines.",
    "Describe a peaceful forest scene in spring.",
    "Summarize the benefits of drinking water.",
    "Explain how to make a simple salad dressing.",
    "Write a kind note to include with a gift of cookies.",
    "Describe how maps help travelers find their way.",
    "Explain what a sonnet is and name one famous example.",
    "Summarize why many people enjoy birdwatching.",
    "Describe the motion of a pendulum clock.",
    "Write a friendly comment for a stranger's puppy photo.",
    "Explain how compost turns food scraps into soil.",
    "List three reasons to keep a gratitude journal.",
    "Describe the appearance of a classic red barn.",
    "Write a short apology for running late to dinner.",
    "Explain how to soft-boil an egg.",
    "Summarize the plot of Goldilocks and the Three Bears.",
    "Describe a cozy winter evening by the fireplace.",
    "Write a one-paragraph appreciation of your favorite season.",
    "Explain how a compass points north.",
    "List three polite ways to disagree in conversation.",
    "Describe how sound travels through the air.",
    "Write a short thank-you for a helpful neighbor.",
    "Explain how to care for a basil plant.",
    "Summarize what makes a sunset visually striking.",
    "Describe a day in the life of a lighthouse keeper.",
    "Write a two-line rhyme about a happy puppy.",
    "Explain how tides are influenced by the moon.",
    "Summarize the plot of The Three Little Pigs.",
    "Describe what makes a library feel welcoming.",
    "Write a kind message to include in a get-well card.",
    "Explain the idea of a food chain with a simple example.",
    "List three gentle stretches for the morning.",
    "Describe the texture of fresh snow underfoot.",
    "Write a short explanation of what a museum does.",
    "Explain how honey is made by bees.",
    "Summarize what makes a good friend.",
    "Describe the feeling of finishing a good book.",
    "Write a one-paragraph description of a farmers' market.",
    "Explain how to fold a paper airplane.",
    "Describe the smell of a pine forest after rain.",
    "Write a short ode to a favorite mug.",
    "Explain why stars appear to twinkle.",
    "Summarize the plot of The Little Red Hen.",
    "Describe a peaceful view from a mountain trail.",
    "Write a short, friendly voicemail greeting."
]


def load_harmless(cache_dir: Path = DATA_DIR, n: int = 100) -> list[str]:
    """Return n harmless instructions. Uses the committed JSON if present, else the seed list."""
    path = cache_dir / "harmless_instructions.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)[:n]
    return _SEEDS[:n]


def write_harmless_json(cache_dir: Path = DATA_DIR) -> Path:
    path = cache_dir / "harmless_instructions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_SEEDS, f, indent=2)
    return path
