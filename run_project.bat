@echo off
echo 1. Generate synthetic dataset
python generate_dataset.py
echo.
echo 2. Train CNN, Decision Tree and Random Forest
python train_models.py
echo.
echo 3. Start web application
streamlit run app.py
pause
