# 🚀 Deployment Guide: Render (Backend) & Vercel (Frontend)

This guide provides simple, step-by-step instructions to deploy the **StudentSignals** system:
- **Backend (Python Flask REST API + Stacking ML Models)** on **[Render](https://render.com/)**
- **Frontend (Tailwind CSS + ApexCharts SPA)** on **[Vercel](https://vercel.com/)**

---

## 🏛️ Deployment Architecture

```
[ User Browser ]
       │
       ├──► (HTML / CSS / JS) ──► Vercel (frontend/)
       │
       └──► (REST API Calls)  ──► Render (backend/ + artifacts/)
```

---

## Part 1: Deploy Backend to Render (Free Tier)

### Option A: Automatic via `render.yaml` (Recommended)
1. Push this project repository to GitHub or GitLab.
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** ➔ **Blueprint**.
4. Select your repository.
5. Render reads `render.yaml` automatically and configures:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `gunicorn backend.app:app`
6. Click **Apply**.
7. Once deployed, copy your Render web service URL (e.g., `https://studentsignals-api.onrender.com`).

### Option B: Manual Web Service Setup on Render
1. In Render Dashboard, click **New +** ➔ **Web Service**.
2. Select your repository.
3. Configure settings:
   - **Name:** `studentsignals-api`
   - **Root Directory:** Leave blank (repository root)
   - **Environment:** `Python 3`
   - **Region:** Any (e.g. Frankfurt, Oregon, Singapore)
   - **Branch:** `main`
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `gunicorn backend.app:app`
   - **Plan:** Free
4. Click **Create Web Service**.
5. When live, copy your backend URL: `https://<your-app-name>.onrender.com`.

---

## Part 2: Deploy Frontend to Vercel (Free Tier)

### Option A: Via Vercel Dashboard (Zero Config)
1. Log in to [Vercel](https://vercel.com/).
2. Click **Add New...** ➔ **Project**.
3. Import your GitHub repository.
4. In the **Configure Project** screen:
   - **Project Name:** `studentsignals`
   - **Framework Preset:** `Other`
   - **Root Directory:** Click **Edit** and select `frontend` (or leave default if using root `vercel.json`).
5. Click **Deploy**.
6. In ~15 seconds, Vercel gives you your live production URL (e.g., `https://studentsignals.vercel.app`).

### Option B: Via Vercel CLI
```bash
# Navigate to frontend folder
cd frontend

# Deploy using Vercel CLI
vercel --prod
```

---

## Part 3: Connect Frontend to Backend

1. Open your live Vercel URL (e.g., `https://studentsignals.vercel.app`).
2. In the top navigation bar, click the **Connect Render API** button:
   ```text
   [ 🟡 Connect Render API ]
   ```
3. Paste your Render backend URL:
   ```text
   https://studentsignals-api.onrender.com
   ```
4. Click **OK**.
5. The status badge will turn green:
   ```text
   [ 🟢 API: Connected (Render) ]
   ```
   *Your choice is saved in `localStorage`, so you never have to re-enter it.*

---

## 💻 Local Development (Run Everything Locally)

```bash
# 1. Start the unified full-stack server
python -m backend.app

# 2. Open in browser:
http://localhost:8000
```
When running on `localhost:8000`, the frontend automatically detects the local backend without any configuration!
