# Bengaluru House prices

Bengaluru house price prediction starter project.

## Project structure

- `data/Bengaluru_House_Data.csv`: source dataset
- `src/preprocessing.py`: data cleaning and feature preparation
- `src/train.py`: train and save model
- `src/evaluate.py`: evaluate trained model
- `src/predict.py`: single prediction from terminal
- `app/app.py`: Streamlit web app
- `models/model.pkl`: trained model artifact

## How to initiate this project

### 1) Activate virtual environment

```powershell
cd "C:\Users\Dell\Desktop\MiniProject\Data_Science_Projects"
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Train model

```powershell
python src\train.py
```

### 4) Evaluate model

```powershell
python src\evaluate.py
```

### 5) Run smoke tests

```powershell
python -m unittest discover -s tests -v
```

### 6) Predict from terminal

```powershell
python src\predict.py --location "Whitefield" --total-sqft 1200 --bath 2 --balcony 1 --bhk 2
```

### 7) Run web app

```powershell
streamlit run app\app.py
```
