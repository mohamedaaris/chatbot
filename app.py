import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, session, Response, redirect, url_for
from werkzeug.utils import secure_filename

from extractor import extract_text_from_pdf, extract_text_from_image, extract_text_from_url, extract_text_from_docx
from train_data import VectorStoreManager
from agentic_bot import AgenticBot
from conversational_ai import ConversationalAI
from agent_tools import AgentTools


# Flask application factory (single-file simple app)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


# Directories for persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
AGENTS_DIR = os.path.join(BASE_DIR, "agents")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(AGENTS_DIR, exist_ok=True)

# Global instances
vector_manager = VectorStoreManager(persist_directory=VECTOR_DIR)
bot = AgenticBot(vector_manager=vector_manager)
conversational_ai = ConversationalAI()
agent_tools = AgentTools(AGENTS_DIR, VECTOR_DIR)


def get_session_id() -> str:
    """Ensure each browser session has an id for separating chat history."""
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    return sid


@app.route("/")
def index():
    """Redirect to chat page."""
    return redirect(url_for('chat_page'))

@app.route("/chat")
def chat_page():
    """Main chat page."""
    sid = get_session_id()
    ai_history = conversational_ai.get_history()
    system_status = agent_tools.get_agent_status()
    return render_template("chat.html", 
                         ai_history=ai_history,
                         system_status=system_status)

@app.route("/create-agent")
def create_agent_page():
    """Create agent page."""
    return render_template("create-agent.html")

@app.route("/my-agents")
def my_agents_page():
    """My agents page."""
    return render_template("my-agents.html")

@app.route("/upload-document")
def upload_document_page():
    """Upload document page."""
    return render_template("upload-document.html")

@app.route("/train-from-url")
def train_from_url_page():
    """Train from URL page."""
    return render_template("train-from-url.html")

@app.route("/add-knowledge")
def add_knowledge_page():
    """Add knowledge page."""
    return render_template("add-knowledge.html")

@app.route("/generate-embed-code")
def generate_embed_code_page():
    """Generate embed code page."""
    return render_template("generate-embed-code.html")

@app.route("/agent-chat")
def agent_chat_page():
    """Agent chat page."""
    return render_template("agent-chat.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF/Image upload and free-form text for training.

    Returns JSON with status messages.
    """
    try:
        # Get agent_id from form data or request
        agent_id = request.form.get("agent_id") or (request.get_json().get("agent_id") if request.get_json() else None)
        
        combined_text_parts: List[str] = []

        # Optional additional plain text provided via textarea
        raw_text = request.form.get("raw_text", "").strip()
        if raw_text:
            combined_text_parts.append(raw_text)

        # Handle file upload (PDF or image)
        if "file" in request.files:
            files = request.files.getlist("file")
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    save_path = os.path.join(UPLOAD_DIR, f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}")
                    file.save(save_path)

                    try:
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in [".pdf"]:
                            extracted = extract_text_from_pdf(save_path)
                        elif ext in [".docx"]:
                            extracted = extract_text_from_docx(save_path)
                        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"]:
                            extracted = extract_text_from_image(save_path)
                        else:
                            # Clean up uploaded file
                            if os.path.exists(save_path):
                                os.remove(save_path)
                            return jsonify({"success": False, "error": "Unsupported file type."}), 400

                        if extracted:
                            combined_text_parts.append(extracted)
                        else:
                            return jsonify({"success": False, "error": "No text found in the uploaded file."}), 400
                            
                    except Exception as e:
                        # Clean up uploaded file
                        if os.path.exists(save_path):
                            os.remove(save_path)
                        return jsonify({"success": False, "error": f"Extraction failed: {str(e)}"}), 500
                    finally:
                        # Clean up uploaded file
                        if os.path.exists(save_path):
                            os.remove(save_path)

        if not combined_text_parts:
            return jsonify({"success": False, "error": "No content provided to train."}), 400

        # Train vector store
        if agent_id:
            # Agent-specific training
            result = agent_tools.train_agent_from_text(agent_id, "\n\n".join(combined_text_parts), "document_upload")
            return jsonify(result)
        else:
            # Global training (legacy)
            text_blob = "\n\n".join(combined_text_parts)
            vector_manager.add_texts([text_blob])
            return jsonify({"success": True, "message": "Training completed successfully."})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/train_url", methods=["POST"])
def train_url():
    """Scrape content from a URL and train a specific agent."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        url = data.get("url", "").strip()
        agent_id = data.get("agent_id")
        
        if not url:
            return jsonify({"success": False, "error": "URL is required"}), 400
        
        if not agent_id:
            return jsonify({"success": False, "error": "Agent ID is required"}), 400

        # Check if agent exists
        agent_result = agent_tools.get_agent_status(agent_id)
        if not agent_result.get("success"):
            return jsonify({"success": False, "error": "Agent not found"}), 404

        # Train the specific agent
        result = agent_tools.train_agent_from_url(agent_id, url)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Answer a user question using RAG over stored knowledge + Llama via Ollama."""
    sid = get_session_id()
    history: List[Tuple[str, str]] = session.get("history", [])

    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "message": "Question is required."}), 400

    try:
        answer, sources = bot.answer_question(question=question, chat_history=history)
        # Update chat history
        history.append((question, answer))
        session["history"] = history
        return jsonify({"ok": True, "answer": answer, "sources": sources, "history": history})
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to generate answer: {e}"}), 500


@app.route("/chat-agent/<agent_id>", methods=["POST"])
def chat_agent(agent_id):
    """Chat with a specific agent."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"success": False, "error": "Question is required"}), 400
        
        # Check if agent exists
        agent_result = agent_tools.get_agent_status(agent_id)
        if not agent_result.get("success"):
            return jsonify({"success": False, "error": "Agent not found"}), 404
        
        agent = agent_result["agent"]
        
        # Get answer using agent-specific knowledge
        answer, sources, emotion_data = bot.answer_question(question=question, agent_id=agent_id)
        
        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources,
            "emotion": emotion_data,
            "agent": {
                "id": agent_id,
                "name": agent["name"],
                "greeting": agent["greeting"]
            }
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/clear", methods=["POST"]) 
def clear():
    """Clear chat history and system messages for the current session."""
    session["history"] = []
    session["system_messages"] = []
    return jsonify({"ok": True})


@app.route("/create_agent", methods=["POST"])
def create_agent():
    """Create a new agent with custom configuration."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        
        name = data.get("name", "").strip()
        greeting = data.get("greeting", "").strip()
        description = data.get("description", "").strip()
        model = data.get("model", "llama3.2:3b")
        
        if not name or not greeting:
            return jsonify({"success": False, "error": "Name and greeting are required"}), 400
        
        # Use agent_tools to create the agent
        result = agent_tools.create_agent(
            name=name,
            greeting=greeting,
            description=description
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """Main conversational endpoint for AgentX-style interaction."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"success": False, "error": "No message provided"}), 400
        
        # Process user input through conversational AI
        response = conversational_ai.process_user_input(user_input, agent_tools)
        
        return jsonify({
            "success": True,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat/history", methods=["GET"])
def get_chat_history():
    """Get conversation history."""
    try:
        history = conversational_ai.get_history()
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat/clear", methods=["POST"])
def clear_chat_history():
    """Clear conversation history."""
    try:
        conversational_ai.clear_history()
        return jsonify({"success": True, "message": "Chat history cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/agents", methods=["GET"])
def list_agents():
    """List all agents."""
    try:
        result = agent_tools.list_agents()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/agents/<agent_id>", methods=["GET"])
def get_agent(agent_id):
    """Get specific agent details."""
    try:
        result = agent_tools.get_agent_status(agent_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/embed_script", methods=["GET", "POST"])
def embed_script():
    """Generate embed script for an agent."""
    if request.method == "POST":
        # Handle POST request with specific agent_id
        data = request.get_json(force=True, silent=True) or {}
        agent_id = data.get("agent_id")
        theme = data.get("theme", "dark")
        
        if agent_id:
            # Get specific agent
            agent_result = agent_tools.get_agent_status(agent_id)
            if agent_result.get("success"):
                agent = agent_result["agent"]
                agent_config = {
                    "name": agent["name"],
                    "greeting": agent["greeting"],
                    "description": agent["description"]
                }
            else:
                return jsonify({"success": False, "error": "Agent not found"}), 404
        else:
            return jsonify({"success": False, "error": "Agent ID is required"}), 400
    else:
        # Handle GET request - get first available agent
        agents_result = agent_tools.list_agents()
        if agents_result.get("success") and agents_result.get("agents"):
            agent = agents_result["agents"][0]
            agent_config = {
                "name": agent["name"],
                "greeting": agent["greeting"],
                "description": agent["description"]
            }
        else:
            agent_config = {
                "name": "AgentX Assistant",
                "greeting": "Hello! I'm AgentX, your AI assistant. How can I help you today?",
                "description": "A powerful AI assistant for creating and managing agents."
            }
    
    script = f"""
<!-- AgentX Embed Script -->
<div id="agentx-widget"></div>
<script>
(function() {{
    const config = {json.dumps(agent_config)};
    const apiUrl = window.location.protocol + '//' + window.location.host;
    
    // Simple widget implementation
    const widget = document.getElementById('agentx-widget');
    widget.innerHTML = `
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
            <div id="chat-toggle" style="background: #5b7cfa; color: white; padding: 15px; border-radius: 50%; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                💬
            </div>
            <div id="chat-window" style="display: none; position: absolute; bottom: 70px; right: 0; width: 350px; height: 500px; background: white; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.12); border: 1px solid #e5e7eb;">
                <div style="padding: 16px; border-bottom: 1px solid #e5e7eb; background: #f9fafb; border-radius: 12px 12px 0 0;">
                    <h3 style="margin: 0; font-size: 16px;">${{config.name}}</h3>
                    <p style="margin: 4px 0 0; font-size: 12px; color: #6b7280;">${{config.description}}</p>
                </div>
                <div id="chat-messages" style="height: 380px; overflow-y: auto; padding: 16px;">
                    <div style="padding: 8px 12px; background: #e9efff; border-radius: 12px; margin-bottom: 8px;">
                        ${{config.greeting}}
                    </div>
                </div>
                <div style="padding: 16px; border-top: 1px solid #e5e7eb;">
                    <input type="text" id="chat-input" placeholder="Ask me anything..." style="width: 100%; padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; outline: none;">
                </div>
            </div>
        </div>
    `;
    
    // Chat functionality
    let isOpen = false;
    document.getElementById('chat-toggle').onclick = () => {{
        isOpen = !isOpen;
        document.getElementById('chat-window').style.display = isOpen ? 'block' : 'none';
    }};
    
    document.getElementById('chat-input').onkeypress = (e) => {{
        if (e.key === 'Enter') {{
            const input = e.target;
            const question = input.value.trim();
            if (!question) return;
            
            // Add user message
            const messages = document.getElementById('chat-messages');
            messages.innerHTML += `<div style="text-align: right; margin: 8px 0;"><div style="display: inline-block; padding: 8px 12px; background: #5b7cfa; color: white; border-radius: 12px;">${{question}}</div></div>`;
            
            input.value = '';
            messages.scrollTop = messages.scrollHeight;
            
            // Send to API
            fetch(apiUrl + '/chat', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ message: question }})
            }})
            .then(res => res.json())
            .then(data => {{
                if (data.success) {{
                    messages.innerHTML += `<div style="margin: 8px 0;"><div style="display: inline-block; padding: 8px 12px; background: #f3f4f6; border-radius: 12px;">${{data.response.content}}</div></div>`;
                }} else {{
                    messages.innerHTML += `<div style="margin: 8px 0;"><div style="display: inline-block; padding: 8px 12px; background: #fee2e2; color: #dc2626; border-radius: 12px;">Error: ${{data.error}}</div></div>`;
                }}
                messages.scrollTop = messages.scrollHeight;
            }})
            .catch(err => {{
                messages.innerHTML += `<div style="margin: 8px 0;"><div style="display: inline-block; padding: 8px 12px; background: #fee2e2; color: #dc2626; border-radius: 12px;">Network error</div></div>`;
                messages.scrollTop = messages.scrollHeight;
            }});
        }}
    }};
}})();
</script>
"""
    
    if request.method == "POST":
        return jsonify({"success": True, "script": script})
    else:
        return script, 200, {'Content-Type': 'text/html'}


@app.route("/agents/<agent_id>", methods=["PUT"])
def update_agent(agent_id):
    """Update a specific agent."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        
        # Check if agent exists
        agent_result = agent_tools.get_agent_status(agent_id)
        if not agent_result.get("success"):
            return jsonify({"success": False, "error": "Agent not found"}), 404
        
        # Update agent data
        agent_data = agent_result["agent"]
        if "name" in data:
            agent_data["name"] = data["name"]
        if "greeting" in data:
            agent_data["greeting"] = data["greeting"]
        if "description" in data:
            agent_data["description"] = data["description"]
        
        agent_data["updated_at"] = datetime.now().isoformat()
        
        # Save updated agent
        agent_tools._save_agent(agent_id, agent_data)
        agent_tools.agents[agent_id] = agent_data
        
        return jsonify({
            "success": True, 
            "message": "Agent updated successfully",
            "agent": agent_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/agents/<agent_id>", methods=["DELETE"])
def delete_agent(agent_id):
    """Delete a specific agent."""
    try:
        # Check if agent exists
        agent_result = agent_tools.get_agent_status(agent_id)
        if not agent_result.get("success"):
            return jsonify({"success": False, "error": "Agent not found"}), 404
        
        # Delete agent from memory and file
        if agent_id in agent_tools.agents:
            del agent_tools.agents[agent_id]
        
        agent_file = os.path.join(AGENTS_DIR, f"{agent_id}.json")
        if os.path.exists(agent_file):
            os.remove(agent_file)
        
        return jsonify({"success": True, "message": "Agent deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge():
    """Add knowledge to an agent's knowledge base."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        agent_id = data.get("agent_id")
        content = data.get("content")
        source = data.get("source", "Manual Entry")
        
        if not agent_id or not content:
            return jsonify({"success": False, "error": "Agent ID and content are required"}), 400
        
        result = agent_tools.train_agent_from_text(agent_id, content, source)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # For development; in production, use a proper WSGI server (gunicorn/uwsgi)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


