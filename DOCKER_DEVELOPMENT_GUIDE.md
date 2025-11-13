# Docker Development Guide for Newbies

**Welcome to Docker development!** This guide will teach you everything you need to know.

---

## 🎯 Why Docker?

### Problems Docker Solves:
❌ **Before**: "Works on my machine" syndrome  
❌ **Before**: Complex setup with multiple terminals  
❌ **Before**: Environment conflicts between projects  
❌ **Before**: Hard to share and deploy  

✅ **With Docker**: Same environment everywhere  
✅ **With Docker**: One command to start everything  
✅ **With Docker**: Isolated environments  
✅ **With Docker**: Easy deployment  

---

## 🚀 Quick Start (2 Commands!)

### Step 1: Setup (One-time)
```bash
# Install Docker Desktop from: https://www.docker.com/products/docker-desktop
# Make sure .env has your GROQ_API_KEY

# Start everything automatically
./dev-start.sh
```

### Step 2: Development (Every day)
```bash
# Terminal 1 - Backend API
docker compose -f docker-compose.dev.yml up backend

# Terminal 2 - Frontend UI  
docker compose -f docker-compose.dev.yml up frontend
```

**That's it!** 🎉

- Frontend: http://localhost:8501
- Backend: http://localhost:8000/docs
- Redis, ChromaDB, Celery Worker run automatically in background

---

## 🧠 Docker Concepts (5-minute crash course)

### 1. **Container** = Lightweight Virtual Machine
```
Your Computer
├── Container 1: Backend API
├── Container 2: Frontend UI  
├── Container 3: Redis Database
├── Container 4: ChromaDB Vector Store
└── Container 5: Celery Worker
```

### 2. **Image** = Blueprint for Container
```bash
# Build image from Dockerfile
docker build -t my-app .

# Run container from image
docker run my-app
```

### 3. **Volume** = Shared Storage
```bash
# Your files ←→ Container files
./data:/app/data        # Share data folder
./app.py:/app/app.py    # Share code files (hot reload!)
```

### 4. **Network** = Container Communication
```bash
# Containers talk to each other by name
frontend → backend:8000  # Not localhost!
backend → redis:6379     # Not localhost!
```

---

## 📁 Development Workflow

### File Structure
```
rag-campus-chatbot/
├── 🐳 Docker files
│   ├── docker-compose.dev.yml     # Development setup
│   ├── Dockerfile.backend.dev     # Backend container
│   ├── Dockerfile.frontend.dev    # Frontend container
│   └── Dockerfile.worker.dev      # Worker container
├── 🚀 Quick scripts
│   ├── dev-start.sh               # Start everything
│   └── dev-stop.sh                # Stop everything
├── 📝 Your code (auto-synced)
│   ├── main.py                    # Backend API
│   ├── app.py                     # Frontend UI
│   ├── rag_pipeline_advanced.py   # RAG logic
│   └── enhanced_document_loader.py # OCR pipeline
└── 📊 Data
    └── data/                      # Documents (auto-synced)
```

### Daily Development Process
```bash
# 1. Start services (once per day)
./dev-start.sh

# 2. Open 2 terminals
# Terminal 1:
docker compose -f docker-compose.dev.yml up backend

# Terminal 2: 
docker compose -f docker-compose.dev.yml up frontend

# 3. Edit files normally in VS Code/IDE
# Files auto-sync to containers
# Services auto-restart on changes ✨

# 4. Test your changes
# Visit http://localhost:8501

# 5. Stop when done
# Ctrl+C in both terminals
./dev-stop.sh
```

---

## 🔧 Development Commands

### Essential Commands
```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs
docker compose -f docker-compose.dev.yml logs backend
docker compose -f docker-compose.dev.yml logs worker
docker compose -f docker-compose.dev.yml logs -f  # Follow logs

# Execute commands inside containers
docker compose -f docker-compose.dev.yml exec backend python -c "print('Hello')"
docker compose -f docker-compose.dev.yml exec worker celery status

# Rebuild containers (after changing Dockerfile)
docker compose -f docker-compose.dev.yml build
docker compose -f docker-compose.dev.yml build backend  # Specific service

# Restart services
docker compose -f docker-compose.dev.yml restart backend
docker compose -f docker-compose.dev.yml restart
```

### Database Commands
```bash
# Check database status
docker compose -f docker-compose.dev.yml exec backend python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/chroma_db')
collection = client.get_collection('collection')
print(f'Chunks: {collection.count()}')
"

# Clear and reingest documents
docker compose -f docker-compose.dev.yml exec worker python -c "
from enhanced_ingestion_worker import clear_collection, process_document
clear_collection.delay().get()
result = process_document.delay('/app/data').get()
print(result)
"

# Manual ingestion
docker compose -f docker-compose.dev.yml exec worker python -c "
from enhanced_ingestion_worker import process_document
task = process_document.delay('/app/data')
print(task.get())
"
```

---

## 📝 How to Modify Code

### 1. Edit Files Normally ✅
```bash
# Just edit in VS Code as usual!
code main.py
code app.py
code rag_pipeline_advanced.py

# Changes auto-sync to containers
# Services auto-restart ✨
```

### 2. Add New Dependencies
```bash
# 1. Edit requirements.txt
echo "new-package==1.0.0" >> requirements.txt

# 2. Rebuild containers
docker compose -f docker-compose.dev.yml build

# 3. Restart services
docker compose -f docker-compose.dev.yml up backend
```

### 3. Add New Files
```bash
# 1. Create file normally
touch new_module.py

# 2. Add volume mapping in docker-compose.dev.yml
# volumes:
#   - ./new_module.py:/app/new_module.py

# 3. Restart service
docker compose -f docker-compose.dev.yml restart backend
```

### 4. Debug Issues
```bash
# Check container status
docker compose -f docker-compose.dev.yml ps

# View logs
docker compose -f docker-compose.dev.yml logs backend

# Enter container for debugging
docker compose -f docker-compose.dev.yml exec backend bash
# Now you're inside the container!
# Run: python, ls, cat, etc.
```

---

## 🛠️ Troubleshooting

### Issue: Containers won't start
```bash
# Check Docker is running
docker info

# Check for port conflicts
netstat -an | grep 8000  # Backend port
netstat -an | grep 8501  # Frontend port

# Clean restart
./dev-stop.sh
docker system prune  # Clean up
./dev-start.sh
```

### Issue: Changes not reflecting
```bash
# 1. Check volume mappings in docker-compose.dev.yml
# 2. Restart specific service
docker compose -f docker-compose.dev.yml restart backend

# 3. If still issues, rebuild
docker compose -f docker-compose.dev.yml build backend
```

### Issue: Database empty
```bash
# Check worker logs
docker compose -f docker-compose.dev.yml logs worker

# Manual ingestion
docker compose -f docker-compose.dev.yml exec worker python -c "
from enhanced_ingestion_worker import process_document
print(process_document.delay('/app/data').get())
"
```

### Issue: Out of disk space
```bash
# Clean up Docker
docker system prune -a  # ⚠️ Removes all unused images
docker volume prune     # ⚠️ Removes unused volumes
```

---

## 🚀 Advanced Development Tips

### Hot Reload Explained
```bash
# Volume mapping enables hot reload:
volumes:
  - ./main.py:/app/main.py  # Your file ←→ Container file

# When you edit ./main.py:
# 1. Change syncs to container instantly
# 2. Uvicorn detects change (--reload flag)
# 3. Service restarts automatically
# 4. No manual restart needed! ✨
```

### Environment Variables
```bash
# Edit .env file normally
GROQ_API_KEY=your_key_here
NEW_SETTING=value

# Restart services to pick up changes
docker compose -f docker-compose.dev.yml restart
```

### Multiple Environments
```bash
# Development (what you're using)
docker compose -f docker-compose.dev.yml up

# Production (for deployment)
docker compose -f docker-compose.yml up

# Testing
docker compose -f docker-compose.test.yml up
```

---

## 🎓 Next Steps

### Week 1: Get Comfortable
- [ ] Use `./dev-start.sh` daily
- [ ] Edit code and see auto-reload
- [ ] Check logs when issues occur
- [ ] Use `docker ps` to see containers

### Week 2: Advanced Usage
- [ ] Try `docker exec` to debug inside containers
- [ ] Add new Python packages
- [ ] Customize docker-compose.dev.yml
- [ ] Create custom Dockerfiles

### Week 3: Production Ready
- [ ] Use production docker-compose.yml
- [ ] Deploy to cloud (AWS, GCP, etc.)
- [ ] Set up CI/CD pipelines
- [ ] Monitor container health

---

## 🆚 Docker vs Local Development

| Aspect | Local Development | Docker Development |
|--------|-------------------|-------------------|
| **Setup** | Multiple terminals, complex | 2 commands |
| **Dependencies** | Virtual env, manual install | Automatic |
| **Consistency** | "Works on my machine" | Same everywhere |
| **Isolation** | Shared system resources | Isolated containers |
| **Sharing** | Hard to share setup | `git clone && ./dev-start.sh` |
| **Deployment** | Different from prod | Same as prod |
| **Learning Curve** | Easy | Medium (worth it!) |

---

## 📚 Resources for Learning More

### Official Docker Docs
- [Docker Tutorial](https://docs.docker.com/get-started/)
- [Docker Compose Tutorial](https://docs.docker.com/compose/gettingstarted/)

### Great YouTube Channels
- [TechWorld with Nana](https://www.youtube.com/c/TechWorldwithNana)
- [Bret Fisher Docker and DevOps](https://www.youtube.com/c/BretFisher)

### Books
- "Docker Deep Dive" by Nigel Poulton
- "Docker in Action" by Jeff Nickoloff

---

## 🎉 You're Ready!

You now know:
✅ How to start development with 2 commands  
✅ How Docker containers work  
✅ How to edit code with hot reload  
✅ How to debug issues  
✅ How to add new features  

**Start developing!** 🚀

```bash
./dev-start.sh

# Terminal 1
docker compose -f docker-compose.dev.yml up backend

# Terminal 2  
docker compose -f docker-compose.dev.yml up frontend

# Open http://localhost:8501 and start building! ✨
```

---

**Happy Docker Development!** 🐳✨
