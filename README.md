# Food Recommendation App

A smart food recommendation system that helps users discover dishes based on their preferences, dietary needs, and caloric requirements. The app leverages machine learning (Logistic Regression) to recommend dishes and can be used both as a web application and via the terminal.

## Features

- **Personalized Recommendations:** Suggests dishes based on user profile, preferred flavors, cuisines, and dish categories.
- **Dietary Awareness:** Each dish is listed with its calories, ingredients, and dietary tags (e.g., vegetarian, gluten-free).
- **Dual Interface:** Accessible as a web application and also runnable in the terminal/command line.
- **Machine Learning Powered:** Uses Logistic Regression for model training to improve recommendation accuracy.

## How It Works

1. **User Profile:** Users input their profile details, including dietary restrictions and preferences.
2. **Search:** Users specify what flavors, cuisines, or dish categories they are interested in.
3. **Recommendations:** The app suggests dishes that best match the user’s input, showing relevant calories, ingredients, and tags.

## Technologies Used

- Python (for backend logic and model training)
- Logistic Regression (machine learning algorithm)
- Streamlit for the web interface
- Command line interface for terminal usage

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kaixxxnnn/food_recommendation_app.git
    cd food_recommendation_app
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

#### Web App

```bash
python scripts/streamlit_app.py
```
Visit `http://localhost:5000` in your browser.

#### Terminal

```bash
python scripts/main.py
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)
