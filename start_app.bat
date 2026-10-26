@echo off
echo ====================================
echo TRACE-Tutor Multimodal Application
echo ====================================
echo.
echo Starting production app...
echo.
echo Features enabled:
echo   - Text, Image, Audio input
echo   - Cross-modal retrieval
echo   - Faithfulness gate
echo   - Clarification agent
echo   - Semantic caching
echo   - Text-to-speech
echo.
echo App will open in browser at http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run app_multimodal.py
