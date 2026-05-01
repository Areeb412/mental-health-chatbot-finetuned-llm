import os
import re
import logging

logger = logging.getLogger(__name__)

# Crisis keywords and phrases (lowercase for matching)
# Grouped by category for maintainability
CRISIS_PATTERNS = [
    # Direct statements
    r'\b(suicide|suicidal|kill myself|end my life|take my life)\b',
    r'\b(want to die|wish i was dead|don\'t want to live)\b',
    r'\b(self.harm|self-harm|cutting myself|hurt myself)\b',
    r'\b(overdose|od\'ing)\b',
    # Hopelessness signals
    r'\b(no point|no reason to live|can\'t go on)\b',
]

DEFAULT_CRISIS_MESSAGE = (
    "I can hear that you're going through something incredibly painful right now, "
    "and I'm really glad you reached out. Please know you are not alone.\n\n"
    "🆘 **Immediate help is available:**\n"
    "• **988 Suicide & Crisis Lifeline:** Call or text **988** (US)\n"
    "• **Crisis Text Line:** Text HOME to **741741**\n"
    "• **International Association for Suicide Prevention:** "
    "https://www.iasp.info/resources/Crisis_Centres/\n\n"
    "Please reach out to one of these services — they have trained counselors "
    "available 24/7 who truly care. 💙"
)


class SafetyLayer:
    """
    Screens user messages for crisis content and returns appropriate resources.
    """

    def __init__(self):
        self.crisis_message = os.getenv(
            "CRISIS_RESPONSE_MSG", DEFAULT_CRISIS_MESSAGE
        )
        # Pre-compile regex patterns for performance
        # Compiling once is much faster than compiling on every call
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in CRISIS_PATTERNS
        ]
        logger.info(f"Safety layer initialized with {len(self.compiled_patterns)} crisis patterns")

    def check(self, message: str) -> str | None:
        """
        Checks a message for crisis content.
        
        Args:
            message: The user's message text
        
        Returns:
            Crisis response string if crisis detected, None otherwise
        """
        for pattern in self.compiled_patterns:
            if pattern.search(message):
                logger.warning(
                    f"Crisis pattern matched: '{pattern.pattern[:40]}...'"
                )
                return self.crisis_message
        return None