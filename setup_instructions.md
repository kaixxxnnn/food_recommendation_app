# Food Recommendation AI - Setup Instructions

### 1. **Create and Activate Virtual Environment**
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. **Install Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 3. **Run the Training Script**
```bash
# Train the model and see sample recommendations
python scripts\main.py
```

### 4. **Launch Web Interface (Optional)**
```bash
# Start the Streamlit web app
streamlit run scripts\streamlit_app.py
```

### 5. **Deactivate Virtual Environment (When Done)**
```bash
# Deactivate the virtual environment when finished
deactivate
```