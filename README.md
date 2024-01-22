# Text Summarization Project

This project is a text summarization application that leverages the Hugging Face BART Large Language Model (LLM). The summarization capabilities are exposed through a user-friendly web application built with Streamlit.

## Overview

The goal of this project is to provide a convenient way for users to input text and receive concise summaries generated by the BART LLM model.

## Project Structure

- `app.py`: Main file containing the Streamlit application code.
- `requirements.txt`: List of project dependencies.
- `data/`: Directory for storing any sample data or models used by the application.

## Getting Started

1. Clone the repository:

    ```bash
    git clone git@github.com:mohiteyashprogrammer/Text_Summarizer_project.git
    cd text-summarization-project
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

4. Open your web browser and navigate to [http://localhost:8501](http://localhost:8501) to access the text summarization application.

## Usage

1. Enter or paste the text you want to summarize in the input field.
2. Click the "Summarize" button.
3. View the generated summary in the output section.

## Dependencies

- `streamlit`: Web application framework for creating interactive apps.
- `transformers`: Library from Hugging Face for state-of-the-art Natural Language Processing.

## Acknowledgments

- Hugging Face for providing the BART Large Language Model.
- Streamlit for the easy-to-use web application framework.

## License

This project is licensed under the [MIT License](LICENSE).

