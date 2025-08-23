@echo off
echo 🚀 Flashcard App - Quick Deployment Setup
echo ================================================
echo.

REM Step 1: Check if .env exists
if exist ".env" (
    echo ✅ .env file found
) else (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Please edit .env file and add your OpenAI API key!
    echo    Open .env and replace 'your-openai-api-key-here' with your actual key
    echo.
    pause
)

REM Step 2: Run pre-deployment check
echo 🔍 Running pre-deployment check...
python pre_deploy_check.py
echo.

REM Step 3: Ask user if they want to test locally
echo 🧪 Would you like to test the app locally first? (y/n)
set /p choice="Enter choice: "

if /i "%choice%"=="y" (
    echo.
    echo 🌟 Starting local test...
    echo Open your browser to: http://localhost:5000
    echo Press Ctrl+C to stop the server
    echo.
    python app.py
) else (
    echo.
    echo 📖 Next steps for deployment:
    echo 1. Make sure your .env file has the correct OPENAI_API_KEY
    echo 2. Push your code to GitHub: git add . ^&^& git commit -m "Ready for deployment" ^&^& git push
    echo 3. Go to render.com and deploy your GitHub repository
    echo 4. Add OPENAI_API_KEY environment variable in Render dashboard
    echo.
    echo 📚 For detailed instructions, see: DEPLOYMENT_STEPS.md
    echo.
)

echo.
echo 🎉 Setup complete! Your app is ready for deployment.
pause
