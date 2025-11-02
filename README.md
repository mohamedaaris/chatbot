# AgentX - AI Agent Platform

An advanced AI web interface that combines a chat-first design with reasoning, automation, and tool integration. Create, manage, and deploy AI agents through natural conversation with Llama 3.2 3B. Built with Flask, featuring a modern dark/light theme interface, real-time chat, and modular agent tools system.

## Features

### 🤖 **Conversational AI Interface**
- **Chat-First Design**: Natural language interaction with Llama 3.2 3B
- **Intelligent Tool Calling**: AI automatically executes backend functions
- **Context Retention**: Remembers conversation history and agent states
- **Real-time Responses**: Streaming chat with typing indicators

### 🛠️ **Agent Management**
- **Create Agents**: "Create a customer service agent named Sarah"
- **Train Agents**: "Train my agent on this document" or "Add knowledge about our products"
- **Deploy Agents**: "Generate embed code for my agent" or "Get a shareable URL"
- **Manage Agents**: "Show me all my agents" or "Update agent settings"

### 🎨 **Modern UI/UX**
- **Dark/Light Themes**: Toggle between themes with persistent preference
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Collapsible Sidebar**: Quick actions and navigation
- **Smooth Animations**: Professional transitions and micro-interactions

### 🔧 **Technical Features**
- **Modular Tools System**: Dynamic backend functions the AI can call
- **Custom Vector Store**: Lightweight, persistent storage with Ollama embeddings
- **Session Persistence**: Chat history and agent states saved between sessions
- **Offline-First**: Runs completely offline except for URL scraping

## Requirements
- Python 3.10+
- Ollama installed and running locally
- Llama model `llama3.2:3b` pulled in Ollama
- Tesseract OCR installed (for image/PDF OCR)

### Windows (PowerShell)
- Install Tesseract: `choco install tesseract` or download from the official site
- Ensure `tesseract.exe` is in PATH

## Install
```bash
# From project root
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Start Ollama
```bash
ollama serve
ollama pull llama3.2:3b
```

## Run the App
```bash
# From project root
set FLASK_SECRET_KEY=replace-me
python app.py
```
Open `http://localhost:5000` in your browser.

## Usage

### 🚀 **Getting Started**
1. **Start Ollama**: `ollama serve` and `ollama pull llama3.2:3b`
2. **Run AgentX**: `python app.py`
3. **Open Browser**: Navigate to `http://localhost:5000`

### 💬 **Chat-First Interaction**
Simply talk to AgentX in natural language:

**Creating Agents:**
- "Create a customer service agent named Sarah"
- "Build a sales assistant that helps with product recommendations"
- "Make a technical support bot for our software"

**Training Agents:**
- "Train my agent on this document"
- "Add knowledge about our company policies"
- "Upload our FAQ document to the knowledge base"

**Deploying Agents:**
- "Generate embed code for my agent"
- "Get a shareable URL for my customer service agent"
- "Make my agent available on our website"

### 🎯 **Quick Actions**
Use the sidebar for common tasks:
- **Create Agent**: Quick agent creation
- **My Agents**: View and manage existing agents
- **Upload Document**: Add training materials
- **Generate Embed**: Get website embed codes
- **Settings**: Configure preferences

### 🔧 **Advanced Features**
- **Dark/Light Mode**: Toggle themes for your preference
- **Chat History**: All conversations are automatically saved
- **Tool Integration**: AI automatically calls backend functions
- **Real-time Updates**: See responses as they're generated

## Project Structure
```
app.py                # Flask server with conversational AI endpoints
conversational_ai.py  # Main AI system for natural language interaction
agent_tools.py        # Modular backend functions for agent management
extractor.py          # PDF/Image OCR and URL text extraction utilities
train_data.py         # Custom vector store manager with Ollama embeddings
agentic_bot.py        # RAG wrapper (retrieve -> prompt -> Ollama)
templates/index.html  # AgentX-style chat-first UI with dark/light themes
requirements.txt      # Python dependencies
```

## Extending AgentX

### 🤖 **Model Integration**
- **Swap Models**: Change `model` parameter in `conversational_ai.py` to use different Ollama models
- **Custom Embeddings**: Modify `train_data.py` to use different embedding models
- **Multi-Model Support**: Add support for multiple AI models in the same session

### 🛠️ **Tool Development**
- **New Tools**: Add functions to `agent_tools.py` for custom agent capabilities
- **API Integration**: Connect external APIs for enhanced agent functionality
- **Custom Actions**: Extend the tool system for specialized use cases

### 🎨 **UI Customization**
- **Theme Development**: Add new themes or customize existing ones in `templates/index.html`
- **Layout Changes**: Modify the chat interface or sidebar layout
- **Mobile Optimization**: Enhance responsive design for specific devices

### 🔧 **Advanced Features**
- **Multi-User Support**: Add user authentication and agent isolation
- **Background Processing**: Implement async training with worker queues
- **Analytics**: Add usage tracking and agent performance metrics
- **Plugin System**: Create a plugin architecture for extensibility

## Troubleshooting
- Tesseract not found: install and add to PATH; on Windows, set `pytesseract.pytesseract_cmd` if needed.
- Ollama connection: ensure `ollama serve` is running; `ollama pull llama3.2:3b` first run may take time.
- Empty answers: verify training succeeded and content was extracted; try asking simpler questions.

## License
MIT
