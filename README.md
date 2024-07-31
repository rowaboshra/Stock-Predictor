
# Stock Market Prediction Model

This project is a stock market prediction application built using Python. It leverages machine learning to predict future stock prices based on historical data.

## Features

- **Stock Data Retrieval**: Fetches historical stock data using the `yfinance` API.
- **Machine Learning Model**: Utilizes a pre-trained Keras model for stock price prediction.
- **Data Visualization**: Visualizes actual stock prices, predicted prices, and moving averages using `matplotlib` and `streamlit`.

## Requirements

- Python 3.6+
- NumPy
- Pandas
- yfinance
- Keras
- TensorFlow
- Streamlit
- Matplotlib
- scikit-learn

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run application.py
    ```
2. Enter a stock symbol (e.g., `GOOG`) to retrieve and display the stock data.
3. The application will show various visualizations, including moving averages and a comparison between actual and predicted prices.

## Model

The model used in this project is a pre-trained neural network saved in the file `Rowa Stock Market Predictor.keras`. It is loaded at runtime to make predictions on the stock data. Here is an example of the use of the `TSLA` Stock. 


<img width="1254" alt="Screenshot 2024-07-31 at 4 34 33 AM" src="https://github.com/user-attachments/assets/b97b5764-c7dd-4d44-8e70-8d3891dff3af">

<img width="1116" alt="Screenshot 2024-07-31 at 4 35 04 AM" src="https://github.com/user-attachments/assets/762495b7-b303-4180-927f-dae65f6baf25">

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: I used resources such as YouTube tutorials to help me with this project.

## Acknowledgements

Special thanks to the developers of the libraries used in this project.

---

Feel free to modify this draft to better suit your project's specifics, especially the description, usage instructions, and any additional features or details.
