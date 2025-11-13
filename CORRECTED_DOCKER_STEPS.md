# CORRECTED Docker Steps

**Issue Fixed**: Dependency conflict in `requirements.txt` ✅  
**Status**: Ready to run on any platform 🚀

---

## 🚀 Step-by-Step Guide (All Platforms)

### Step 0: Prerequisites
- ✅ Docker Desktop installed and running
- ✅ Your GROQ API key ready

### Step 1: Setup (One-time)
```bash
# Clone/navigate to project
cd rag-campus-chatbot

# Setup environment file
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_actual_key_here

# Start background services
# macOS/Linux:
./dev-start.sh

# Windows:
dev-start.bat
```

### Step 2: Run Development (Daily)

**⚠️ IMPORTANT: Do NOT activate virtual environment!**

```bash
# Open 2 terminals in the project folder

# Terminal 1 - Backend API:
docker compose -f docker-compose.dev.yml up backend

# Terminal 2 - Frontend UI:
docker compose -f docker-compose.dev.yml up frontend
```

### Step 3: Access
- Frontend UI: http://localhost:8501
- Backend API: http://localhost:8000/docs

---

## ❌ What You Did Wrong

### Mistake 1: Virtual Environment
```bash
❌ source venv/bin/activate  # DON'T DO THIS with Docker!
❌ docker compose up backend

✅ docker compose -f docker-compose.dev.yml up backend  # Just this
```

### Mistake 2: Requirements Conflict
- Had duplicate `tqdm` versions in requirements.txt
- **Fixed**: ✅ Removed duplicate entry

---

## 🌍 Cross-Platform Instructions

### Windows
```cmd
REM Install Docker Desktop for Windows
REM Clone project
git clone <your-repo>
cd rag-campus-chatbot

REM Setup
copy .env.example .env
REM Edit .env in Notepad: GROQ_API_KEY=your_key

REM Start services
dev-start.bat

REM Open 2 Command Prompts:
REM Terminal 1:
docker compose -f docker-compose.dev.yml up backend

REM Terminal 2:
docker compose -f docker-compose.dev.yml up frontend
```

### Linux (Ubuntu/Debian)
```bash
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose-plugin
sudo systemctl start docker
sudo usermod -aG docker $USER  # Logout/login after this

# Clone project
git clone <your-repo>
cd rag-campus-chatbot

# Setup
cp .env.example .env
nano .env  # Add GROQ_API_KEY=your_key

# Start services
chmod +x dev-start.sh
./dev-start.sh

# Open 2 terminals:
# Terminal 1:
docker compose -f docker-compose.dev.yml up backend

# Terminal 2:
docker compose -f docker-compose.dev.yml up frontend
```

### macOS (What you have)
```bash
# Install Docker Desktop from website
# Clone project
cd rag-campus-chatbot

# Setup
cp .env.example .env
nano .env  # Add GROQ_API_KEY=your_key

# Start services
./dev-start.sh

# Open 2 terminals:
# Terminal 1 (NO virtual env!):
docker compose -f docker-compose.dev.yml up backend

# Terminal 2 (NO virtual env!):
docker compose -f docker-compose.dev.yml up frontend
```

---

## 🧪 Test Your Fix

Run this exactly (without virtual env):

```bash
# 1. Clean any existing containers
docker compose -f docker-compose.dev.yml down

# 2. Build fresh (should work now)
docker compose -f docker-compose.dev.yml build

# 3. Start background services
docker compose -f docker-compose.dev.yml up -d redis chroma worker

# 4. Test backend (Terminal 1)
docker compose -f docker-compose.dev.yml up backend

# 5. Test frontend (Terminal 2 - open new terminal)
docker compose -f docker-compose.dev.yml up frontend
```

**Expected Results**:
- ✅ No dependency conflicts
- ✅ Backend starts on http://localhost:8000
- ✅ Frontend starts on http://localhost:8501
- ✅ Can ask questions and get answers

---

## 🎯 Key Takeaways

### DO ✅
1. **Use Docker without virtual env**
2. **Same commands work on Windows/macOS/Linux**
3. **Edit .env file with your API key**
4. **Run `./dev-start.sh` or `dev-start.bat` first**

### DON'T ❌
1. **Don't activate virtual environment with Docker**
2. **Don't mix local and Docker development**
3. **Don't skip the setup script**
4. **Don't forget to start Docker Desktop**

---

## 🚀 Ready to Try Again

Your system is now fixed and ready for cross-platform deployment! 

Try the corrected steps above and it should work perfectly. 🎉
