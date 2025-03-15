# Document Analysis Assistant

This application allows you to upload various document types (PDF, DOCX, CSV, XLS, EPUB, DJVU), process them, and interact with them in multiple ways:

1. **Chat with your documents**: Ask questions about the content of your documents
2. **Generate summaries**: Create concise summaries of your document content
3. **Visualize data**: Create charts and graphs from your data files (CSV, Excel)
4. **Local Data**: No need to upload your to internet, every local stays local.

## Features

- Multi-document support (PDF, DOCX, CSV, XLS, EPUB, DJVU)
- Text extraction and processing
- Vector embeddings for semantic search
- Conversational AI interface
- Document summarization
- Data visualization
- Integration with both Hugging Face models and local Ollama


## Getting Started

### Installation Instructions

1. Clone this repository. Save all the files to your project directory.
2. Go to your dir "DocsChat" in your computer.
3. Run virtual environment of your preference, in this case I´m using `venv`
   **Create Environment**:
    ```sh
    python -m venv DocsChat
    ```
    - **Activate**:
      - **Windows**:
        ```sh
        DocsChat\Scripts\activate
        ```
      - **Linux/macOS**:
        ```sh
        source DocsChat/bin/activate
        ```
    - **Deactivate**:
      ```sh
      deactivate

4. Install the required dependencies:

    ```shellscript
    pip install -r requirements.txt
    ```


5. (Optional) Set up Ollama locally:

    - Install Ollama from [https://ollama.ai/](https://ollama.ai/)
    - Pull the llama3 model: `ollama pull llama3`

6.  Edit file `.env.example`  with your Hugging Face API token:
    ```plaintext
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```
    rename `.env.example` into `.env`
    ```shellscript
    cp .env.example .env
    ```





7. Run the application:

    ```shellscript
    streamlit run app.py
    ```




### Using the Application

1. **Upload Documents**: Use the sidebar to upload your documents (PDF, DOCX, CSV, Excel)
2. **Process Documents**: Click the "Process Documents" button to extract and analyze the content
3. **Chat**: Ask questions about your documents in the Chat tab
4. **Summarize**: Generate document summaries in the Summarize tab
5. **Visualize**: Create charts from your data files in the Visualize tab


### Model Selection

You can choose between:

- **Hugging Face models**: Uses the google/flan-t5-xxl model by default
- **Local Ollama**: Check the "Use Local Ollama" box to use your local Llama2 model

## Next Steps

1. **Fine-tune the models**: You might want to fine-tune the models for better performance on your specific documents
2. **Add more visualization options**: Expand the visualization capabilities for more complex data analysis
3. **Improve the UI**: Enhance the user interface for a better user experience
4. **Add authentication**: If you plan to deploy this, consider adding user authentication

Support page:
- Paypal https://paypal.me/IrfanHandrian
- Buy me Coffee https://buymeacoffee.com/handrianirv
