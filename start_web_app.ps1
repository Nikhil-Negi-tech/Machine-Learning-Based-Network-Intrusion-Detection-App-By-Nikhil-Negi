Write-Host "🚀 Starting AI Network Intrusion Detection Web App..." -ForegroundColor Green
Write-Host ""
Write-Host "📡 The web app will open in your default browser" -ForegroundColor Cyan
Write-Host "🌐 Usually at: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 To stop the app, press Ctrl+C in this window" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
streamlit run web_app.py