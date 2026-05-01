import os
import logging
import random
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

from src.inference import EmpathyBot
from src.safety import SafetyLayer
from src.logger import setup_logger

# Load environment variables from .env file
# Must be called BEFORE any os.getenv() calls
load_dotenv()

# ─── Application Setup ──────────────────────────────────────
app = Flask(__name__)

# Set up structured logging — crucial for debugging in production
logger = setup_logger("app")

# ─── Initialize Model (loaded once at startup, not per request) ──────────────
# Loading a model takes 5-30 seconds. We do it once when the server starts,
# then reuse the loaded model for all subsequent requests.
logger.info("Initializing models...")
MODEL_PATH = os.getenv("MODEL_PATH", "models/empathy-model")

try:
    bot = EmpathyBot(model_path=MODEL_PATH)
    logger.info("✅ Fine-tuned model loaded successfully")
except (FileNotFoundError, Exception) as e:
    logger.warning(
        "Using demo mode because the fine-tuned model could not be loaded. "
        f"Reason: {e}. Run: uv run python train/finetune.py"
    )
    bot = None

safety = SafetyLayer()
logger.info("✅ Safety layer initialized")


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    Serves the main chat page.
    
    render_template() looks in the 'templates/' folder for 'chat.html'
    and returns it as an HTTP response. Jinja2 processes any {{ }} variables.
    """
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Main chat endpoint — receives user message, returns bot response.
    
    Request format (JSON):
        { "msg": "I've been feeling really anxious lately" }
    
    Response format (JSON):
        { "response": "I hear you. Anxiety can be really overwhelming..." }
    
    Why POST and not GET?
        GET requests are logged in browser history and server logs.
        We use POST to keep the user's sensitive messages more private.
    """
    # 1. Extract the message from the request body
    data = request.get_json(silent=True)
    
    if not data or "msg" not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = str(data["msg"]).strip()
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    logger.info(f"Received message (length: {len(user_message)} chars)")

    # 2. Run safety check BEFORE generating response
    # This catches crisis language (suicidal ideation, self-harm) and
    # returns an appropriate crisis helpline message instead of AI text
    crisis_response = safety.check(user_message)
    if crisis_response:
        logger.warning("Crisis language detected — returning safety response")
        return jsonify({"response": crisis_response, "is_crisis": True})

    # 3. Generate response (with fallback demo mode)
    try:
        if bot is None:
            # Demo mode: use empathetic template responses
            demo_responses = [
                "I hear you. That sounds really challenging. Could you tell me more?",
                "Thank you for sharing that with me. How are you coping with this?",
                "I'm listening. It's important that you're expressing how you feel.",
                "That must be difficult. You're not alone in feeling this way.",
                "I appreciate your openness. What's been the hardest part for you?",
                "That's a lot to handle. I'm here to listen and support you.",
            ]
            response = random.choice(demo_responses)
            logger.info("Using demo response (model training in progress)")
        else:
            response = bot.generate(user_message)
        
        logger.info(f"Response sent (length: {len(response)} chars)")
        return jsonify({"response": response, "is_crisis": False})
    
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({
            "response": "I'm here for you. Please tell me more about what you're experiencing."
        })


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for Docker, Kubernetes, and load balancers.
    Returns 200 if the app is running, 503 if the model failed to load.
    """
    if bot is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "degraded", "model_loaded": False}), 503


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       🧠 Mental Health Support Chatbot                   ║
║       Running at http://localhost:{port}                   ║
║       Press CTRL+C to stop                               ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host="0.0.0.0", port=port, debug=debug)