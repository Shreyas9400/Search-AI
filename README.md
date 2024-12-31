# Search-AI

This application uses Gradio, SearxNG, and Groq's Llama 3.1 model (or Ollama) to create a web scraper that can answer questions by searching the internet or using its knowledge base.

## Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd web_scraper_app
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate    # On Windows
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `.env` file:**

    Create a file named `.env` in the root directory of the project and add your Groq API keys and Ollama host (if using) like this:

    ```
    GROQ_API_KEY_1=<your_groq_api_key_1>
    GROQ_API_KEY_2=<your_groq_api_key_2>
    OLLAMA_HOST=<your_ollama_host_and_port> # e.g., localhost:11434
    ```

    Replace `<your_groq_api_key_1>`, `<your_groq_api_key_2>`, and `<your_ollama_host_and_port>` with your actual API keys and Ollama host details.

## Running the Application

1. **Make sure you are in the virtual environment:**

    ```bash
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate    # On Windows
    ```

2. **Run the application:**

    ```bash
    python app.py
    ```

3. **Access the application:**

    The application will be accessible in your web browser at the URL provided in the console output (usually `http://127.0.0.1:7860`).

## Additional Notes

*   **SearxNG:** This application is designed to work with a SearxNG instance. You'll need to have a SearxNG instance running and accessible. You can either set up your own or use a public instance. The `base_url` in the `ChatBot` class of `app.py` needs to be updated to point to your SearxNG instance.
*   **Groq API Keys:** You need to provide your Groq API keys in the `.env` file for the application to work.
*   **Ollama:** If you are using Ollama, ensure it is running and accessible at the specified host and port.
