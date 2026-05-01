# ✅ Getting Started Guide - Mental Health Chatbot

This guide walks you through running the Mental Health Support Chatbot locally in **3 easy steps**.

---

## 📋 What You'll Need

- ✅ **Python 3.12+** (check with `python --version`)
- ✅ **uv** package manager ([download here](https://docs.astral.sh/uv/getting-started/installation/))
- ✅ **5 minutes** of time
- ✅ **Optional**: GPU for faster training (or use free Google Colab)

---

## 🚀 Step 1: Setup (2 minutes)

### Clone the Repository
```bash
git clone <repository-url>
cd mental-health-chatbot-finetuned-llm
```

### Install Dependencies
```bash
uv sync
```

This command:
- ✅ Creates a Python virtual environment
- ✅ Installs all required packages (Flask, PyTorch, Transformers, etc.)
- ✅ Sets up the project for local development

**Expected output:**
```
Resolved 96 packages in 2ms
Prepared virtual environment
```

---

## 🎬 Step 2: Run the App (1 minute)

### Start the Chat Server

**On Windows:**
```bash
./run.bat
```

**On Mac/Linux:**
```bash
uv run python app.py
```

### You'll See:
```
╔══════════════════════════════════════════════════════════╗
║       🧠 Mental Health Support Chatbot                   ║
║       Running at http://localhost:5000                   ║
║       Press CTRL+C to stop                               ║
╚══════════════════════════════════════════════════════════╝

* Running on http://127.0.0.1:5000
```

The app is running! 🎉

---

## 💬 Step 3: Open the Chat UI

Open your web browser and visit:
```
http://localhost:5000
```

You'll see a beautiful chat interface. Start typing messages!

**Example conversation:**
- User: "I've been feeling really anxious lately"
- Bot: "I hear you. Anxiety can feel overwhelming. What's been triggering it?"

---

## 🎓 Optional: Train Your Own Model (Advanced)

To use a fine-tuned model instead of demo responses:

### 1. Download Dataset
```bash
uv run python data/prepare_data.py
```

This downloads 25k+ empathetic conversations (~5 minutes)

### 2. Fine-Tune Model
```bash
uv run python train/finetune.py
```

**Time required:**
- 🔧 CPU: ~45 minutes
- ⚡ GPU: ~5-10 minutes (recommended!)
- ☁️ Google Colab (free): ~15 minutes

### 3. Restart App
Once training completes, restart the app:
```bash
uv run python app.py
```

The app will automatically load your trained model!

---

## 🧪 Verify Everything Works

### Test Endpoint 1: Homepage
```bash
curl http://localhost:5000/
# Should return HTML (chat interface)
```

### Test Endpoint 2: Chat
```bash
curl -X POST http://localhost:5000/get \
  -H "Content-Type: application/json" \
  -d '{"msg": "I feel anxious"}'
# Should return: {"response": "...", "is_crisis": false}
```

### Test Endpoint 3: Health Check
```bash
curl http://localhost:5000/health
# Should return: {"status": "degraded", "model_loaded": false}
# (degraded = demo mode, no trained model yet)
```

---

## 🐛 Common Issues & Solutions

### Issue: "uv: command not found"
**Solution**: Install uv from https://docs.astral.sh/uv/getting-started/installation/

### Issue: "Port 5000 already in use"
**Solution**: 
```bash
# Use a different port
FLASK_PORT=5001 uv run python app.py

# Or kill the existing process
# Windows: netstat -ano | findstr :5000
# Mac/Linux: lsof -i :5000
```

### Issue: "No module named 'flask'"
**Solution**:
```bash
uv sync --force
uv run python app.py
```

### Issue: "App runs but model not loaded"
**Solution**: This is normal in demo mode. Train a model:
```bash
uv run python data/prepare_data.py
uv run python train/finetune.py
```

---

## 📊 Understanding the Modes

### Demo Mode (Running Now)
- ✅ App starts immediately
- ✅ Responses use empathetic templates
- ✅ Perfect for testing UI/API
- ❌ Model not loaded (no AI generation)

### Production Mode (After Training)
- ✅ Fine-tuned model loaded
- ✅ Real AI-generated responses
- ✅ Crisis detection active
- ⏱️ Takes 5-45 minutes to train

---

## 🌐 Access the App

| Feature | URL |
|---------|-----|
| Chat Interface | http://localhost:5000 |
| API Endpoint | POST http://localhost:5000/get |
| Health Check | http://localhost:5000/health |

---

## 📚 Next Steps

1. **Explore the Code**: Check `app.py`, `src/inference.py`, `src/safety.py`
2. **Train a Model**: Follow the optional training steps above
3. **Deploy**: See Dockerfile for containerization
4. **Customize**: Edit response behavior in `src/inference.py`

---

## 🎓 Understanding the Architecture

```
User Input (Web UI)
     ↓
Flask Server (app.py)
     ├─ Safety Check (src/safety.py)
     │  └─ Detects crisis language
     ├─ Model Inference (src/inference.py)
     │  └─ Generates response
     └─ Return JSON Response
     ↓
Display in Chat UI
```

---

## ⚠️ Important Notes

- ⚠️ This chatbot is **educational** and **not** a replacement for professional mental health care
- 🔒 All processing happens **locally** - no data sent to external servers
- 🚨 For real emergencies: **Call 988** (US) or contact local emergency services

---

## 🤝 Need Help?

- **Check README.md** for comprehensive documentation
- **Read training guide** in mental_health_chatbot_guide.md
- **Open GitHub issues** for bugs or feature requests
- **Review troubleshooting** section in main README

---

## ✅ You're All Set!

You now have a working Mental Health Support Chatbot running locally. 🎉

**Next time you want to run it:**
```bash
cd mental-health-chatbot-finetuned-llm
uv run python app.py
# or: ./run.bat (Windows)
```

**Questions?** Check the README.md or GitHub issues.

**Happy coding!** ❤️
