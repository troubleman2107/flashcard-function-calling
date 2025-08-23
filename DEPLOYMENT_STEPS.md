# 🚀 Step-by-Step Deployment Guide

## 📋 Pre-Deployment Checklist

### Step 1: Run Pre-Deployment Check
```bash
python pre_deploy_check.py
```

**Expected Output:**
```
✅ PASS - Python Version
✅ PASS - Dependencies  
✅ PASS - Environment
✅ PASS - Required Files
🎉 ALL CHECKS PASSED! Ready for deployment!
```

If any checks fail, follow the fixes below before proceeding.

---

## 🔧 Fix Common Issues

### ❌ Dependencies Missing
```bash
pip install -r requirements.txt
```

### ❌ Environment Issues  
1. Create `.env` file in your project root
2. Add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### ❌ Missing Files
Ensure these files exist:
- `app.py` ✅
- `requirements.txt` ✅  
- `templates/chat.html` ✅
- `static/styles.css` ✅

---

## 🌐 Deployment Options

### Option 1: Render.com (FREE - Recommended for beginners)

#### Prerequisites:
- GitHub account
- Code pushed to GitHub repository

#### Steps:

**Step 1: Prepare Your Repository**
```bash
# Add all files
git add .

# Commit changes
git commit -m "Ready for deployment"

# Push to GitHub
git push origin main
```

**Step 2: Deploy on Render**
1. Go to https://render.com
2. Sign up with your GitHub account
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Select your flashcard repository

**Step 3: Configure Service**
- **Name**: `flashcard-app` (or your preferred name)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

**Step 4: Add Environment Variables**
- Click "Advanced" → "Environment Variables"
- Add: 
  - Key: `OPENAI_API_KEY`
  - Value: `your-actual-openai-api-key`

**Step 5: Deploy**
- Click "Create Web Service"
- Wait 5-10 minutes for deployment
- Your app will be live at: `https://your-app-name.onrender.com`

---

### Option 2: Railway.app (Simple Alternative)

#### Steps:
1. Go to https://railway.app
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Add environment variable:
   - `OPENAI_API_KEY=your-api-key`
6. Deploy automatically
7. Access your app via provided URL

---

### Option 3: Heroku (Paid)

#### Prerequisites:
- Heroku CLI installed
- Credit card for account verification

#### Steps:
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Login to Heroku:
   ```bash
   heroku login
   ```
3. Create new app:
   ```bash
   heroku create your-app-name
   ```
4. Set environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your-api-key
   ```
5. Deploy:
   ```bash
   git push heroku main
   ```
6. Open your app:
   ```bash
   heroku open
   ```

---

## 📁 Required Deployment Files

Your project should have these files:

```
flashcard-function-calling/
├── app.py                 ✅ Main application
├── requirements.txt       ✅ Dependencies
├── .env                  ✅ Local environment (don't commit)
├── .env.example          📝 Environment template  
├── Procfile              📝 For Heroku
├── render.yaml           📝 For Render.com
├── templates/
│   ├── chat.html         ✅ Main chat interface
│   ├── index.html        ✅ Landing page
│   └── ...
├── static/
│   ├── styles.css        ✅ Styles
│   └── audio/            ✅ Audio files
└── chroma_db/            ✅ Vector database
```

Let me create the missing deployment files for you:

---

## 🛠️ Creating Missing Deployment Files

### Create Procfile (for Heroku):
```bash
echo "web: python app.py" > Procfile
```

### Create render.yaml (for Render.com):
```yaml
services:
  - type: web
    name: flashcard-app
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### Create .env.example:
```
# Copy this to .env and fill in your values
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
PORT=5000
```

---

## ✅ Post-Deployment Testing

After deployment, test these features:

1. **Basic Access**: Can you open the app URL?
2. **Chat Interface**: Does the chat interface load?
3. **Vocabulary Search**: Try "tìm từ liên quan đến du lịch"
4. **Audio Features**: Do audio pronunciations work?
5. **Error Handling**: Test with invalid inputs

---

## 🚨 Troubleshooting

### App Won't Start
- Check deployment logs
- Verify environment variables are set
- Ensure requirements.txt has correct versions

### 500 Internal Server Error
- Missing OPENAI_API_KEY environment variable
- Database initialization issues
- Check application logs

### Dependencies Issues
- Update requirements.txt versions
- Clear build cache and redeploy

### Audio Not Working
- Check if audio files are included in deployment
- Verify static file serving is configured

---

## 💰 Cost Breakdown

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| **Render.com** | ✅ 512MB RAM, 500 build minutes | $7/month for more resources |
| **Railway** | $5 credit monthly | $5+/month usage-based |
| **Heroku** | ❌ No free tier | $7/month minimum |

**Recommendation**: Start with Render.com's free tier.

---

## 🎉 Success Checklist

- [ ] Pre-deployment check passes
- [ ] Code pushed to GitHub
- [ ] Deployment platform chosen
- [ ] Environment variables configured  
- [ ] App deployed successfully
- [ ] All features tested and working
- [ ] Custom domain configured (optional)

---

## 📞 Getting Help

If you encounter issues:

1. **Check the logs** on your deployment platform
2. **Run pre-deployment check** locally
3. **Verify environment variables** are set correctly
4. **Test locally first** with `python app.py`

Your flashcard app is ready for the world! 🌍✨
