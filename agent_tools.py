"""
Agent Tools system for AgentX-style interface.
Provides modular backend functions that the AI can call dynamically.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from extractor import extract_text_from_pdf, extract_text_from_image, extract_text_from_url
from train_data import VectorStoreManager
from agentic_bot import AgenticBot


class AgentTools:
    """Modular tools system for agent management and operations."""
    
    def __init__(self, base_dir: str = "agents", vector_dir: str = "vector_store"):
        self.base_dir = base_dir
        self.vector_dir = vector_dir
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)
        
        # Initialize vector store manager
        self.vector_manager = VectorStoreManager(persist_directory=vector_dir)
        
        # Load existing agents
        self.agents = self._load_agents()
        
    def _load_agents(self) -> Dict[str, Dict[str, Any]]:
        """Load all existing agents from disk."""
        agents = {}
        if os.path.exists(self.base_dir):
            for filename in os.listdir(self.base_dir):
                if filename.endswith('.json'):
                    agent_id = filename[:-5]  # Remove .json extension
                    try:
                        with open(os.path.join(self.base_dir, filename), 'r') as f:
                            agents[agent_id] = json.load(f)
                    except Exception as e:
                        print(f"Error loading agent {agent_id}: {e}")
        return agents
    
    def _save_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> None:
        """Save agent data to disk."""
        agent_file = os.path.join(self.base_dir, f"{agent_id}.json")
        with open(agent_file, 'w') as f:
            json.dump(agent_data, f, indent=2)
    
    def create_agent(self, name: str, greeting: str, description: str = "") -> Dict[str, Any]:
        """Create a new agent with specified configuration."""
        try:
            # Generate unique agent ID
            agent_id = str(uuid.uuid4())
            agent_data = {
                'id': agent_id,
                'name': name,
                'greeting': greeting,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'knowledge_base_status': 'empty'
            }
            
            # Save agent
            self._save_agent(agent_id, agent_data)
            self.agents[agent_id] = agent_data
            
            return {
                "success": True,
                "message": f"Created agent '{name}' successfully!",
                "agent_id": agent_id,
                "agent": agent_data
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create agent: {str(e)}"}

    def create_agent_from_conversation(self, user_input: str) -> Dict[str, Any]:
        """Create a new agent based on conversational input."""
        try:
            # Extract agent details from user input
            agent_data = self._parse_agent_creation_input(user_input)
            
            # Generate unique agent ID
            agent_id = str(uuid.uuid4())
            agent_data['id'] = agent_id
            agent_data['created_at'] = datetime.now().isoformat()
            agent_data['updated_at'] = datetime.now().isoformat()
            
            # Save agent
            self._save_agent(agent_id, agent_data)
            self.agents[agent_id] = agent_data
            
            return {
                "success": True,
                "message": f"Created agent '{agent_data['name']}' successfully!",
                "agent_id": agent_id,
                "agent_data": agent_data
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create agent: {str(e)}"}
    
    def _parse_agent_creation_input(self, user_input: str) -> Dict[str, Any]:
        """Parse user input to extract agent configuration."""
        # Default values
        agent_data = {
            "name": "My Agent",
            "greeting": "Hello! I'm your AI assistant. How can I help you today?",
            "description": "A helpful AI assistant trained on your knowledge base.",
            "personality": "Helpful and professional",
            "model": "llama3.2:3b"
        }
        
        # Simple parsing - in a real implementation, you'd use more sophisticated NLP
        user_input_lower = user_input.lower()
        
        # Extract name
        name_patterns = [
            r"name.*?['\"](.*?)['\"]", r"call.*?['\"](.*?)['\"]", 
            r"named.*?['\"](.*?)['\"]", r"agent.*?['\"](.*?)['\"]"
        ]
        for pattern in name_patterns:
            import re
            match = re.search(pattern, user_input_lower)
            if match:
                agent_data["name"] = match.group(1).strip()
                break
        
        # Extract greeting
        greeting_patterns = [
            r"greeting.*?['\"](.*?)['\"]", r"say.*?['\"](.*?)['\"]",
            r"welcome.*?['\"](.*?)['\"]", r"hello.*?['\"](.*?)['\"]"
        ]
        for pattern in greeting_patterns:
            import re
            match = re.search(pattern, user_input_lower)
            if match:
                agent_data["greeting"] = match.group(1).strip()
                break
        
        # Extract description
        desc_patterns = [
            r"description.*?['\"](.*?)['\"]", r"about.*?['\"](.*?)['\"]",
            r"describe.*?['\"](.*?)['\"]", r"purpose.*?['\"](.*?)['\"]"
        ]
        for pattern in desc_patterns:
            import re
            match = re.search(pattern, user_input_lower)
            if match:
                agent_data["description"] = match.group(1).strip()
                break
        
        return agent_data
    
    def list_agents(self) -> Dict[str, Any]:
        """List all available agents."""
        if not self.agents:
            return {
                "success": True,
                "message": "No agents found. Create your first agent to get started!",
                "agents": []
            }
        
        agent_list = []
        for agent_id, agent_data in self.agents.items():
            agent_list.append({
                "id": agent_id,
                "name": agent_data.get("name", "Unnamed Agent"),
                "description": agent_data.get("description", "No description"),
                "greeting": agent_data.get("greeting", "Hello!"),
                "created_at": agent_data.get("created_at", "Unknown"),
                "knowledge_base_status": agent_data.get("knowledge_base_status", "empty"),
                "status": "active"
            })
        
        return {
            "success": True,
            "message": f"Found {len(agent_list)} agent(s)",
            "agents": agent_list
        }

    def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """Get status of a specific agent or all agents."""
        if agent_id:
            if agent_id in self.agents:
                return {
                    "success": True,
                    "agent": self.agents[agent_id]
                }
            else:
                return {
                    "success": False,
                    "error": "Agent not found"
                }
        else:
            # Return general status
            return {
                "success": True,
                "status": "active",
                "total_agents": len(self.agents)
            }

    def train_agent_from_text(self, agent_id: str, text_content: str, source: str = "user_input") -> Dict[str, Any]:
        """Train an agent with text content."""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}
            
            # Create agent-specific vector store directory
            agent_vector_dir = os.path.join(self.vector_dir, f"agent_{agent_id}")
            os.makedirs(agent_vector_dir, exist_ok=True)
            
            # Create agent-specific vector manager
            from train_data import VectorStoreManager
            agent_vector_manager = VectorStoreManager(persist_directory=agent_vector_dir)
            
            # Add text to agent-specific vector store
            agent_vector_manager.add_texts(
                texts=[text_content], 
                metadatas=[{"source": source}]
            )
            
            # Update agent status
            self.agents[agent_id]["knowledge_base_status"] = "trained"
            self.agents[agent_id]["updated_at"] = datetime.now().isoformat()
            self.agents[agent_id]["vector_store_path"] = agent_vector_dir
            self._save_agent(agent_id, self.agents[agent_id])
            
            return {
                "success": True,
                "message": f"Agent '{self.agents[agent_id]['name']}' trained successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def train_agent_from_url(self, agent_id: str, url: str) -> Dict[str, Any]:
        """Train an agent with content from a URL."""
        try:
            if agent_id not in self.agents:
                return {"success": False, "error": "Agent not found"}
            
            # Extract content from URL
            from extractor import extract_text_from_url
            content = extract_text_from_url(url)
            
            if not content:
                return {"success": False, "error": "No content found at URL"}
            
            # Train with extracted content
            return self.train_agent_from_text(agent_id, content, f"url:{url}")
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def train_agent_from_conversation(self, user_input: str) -> Dict[str, Any]:
        """Train an agent based on conversational input."""
        try:
            # For now, we'll train the global vector store
            # In a full implementation, you'd select which agent to train
            
            if "upload" in user_input.lower() or "document" in user_input.lower():
                return {
                    "success": True,
                    "message": "To upload documents, please use the file upload interface in the sidebar.",
                    "action": "upload_interface"
                }
            elif "url" in user_input.lower() or "website" in user_input.lower():
                return {
                    "success": True,
                    "message": "To train from a URL, please provide the URL in the sidebar interface.",
                    "action": "url_training"
                }
            else:
                return {
                    "success": True,
                    "message": "I can help you train your agents. You can upload documents, add text, or train from websites. What would you like to do?",
                    "action": "clarify_training"
                }
        except Exception as e:
            return {"success": False, "error": f"Training error: {str(e)}"}
    
    def deploy_agent_from_conversation(self, user_input: str) -> Dict[str, Any]:
        """Deploy an agent based on conversational input."""
        try:
            if "embed" in user_input.lower() or "website" in user_input.lower():
                return {
                    "success": True,
                    "message": "I can generate embed codes for your agents. Use the Deploy section in the sidebar to generate embed codes.",
                    "action": "generate_embed"
                }
            elif "share" in user_input.lower() or "public" in user_input.lower():
                return {
                    "success": True,
                    "message": "I can provide shareable URLs for your agents. Use the Deploy section to get share links.",
                    "action": "get_share_url"
                }
            else:
                return {
                    "success": True,
                    "message": "I can help you deploy your agents. You can generate embed codes for websites or get shareable URLs. What type of deployment do you need?",
                    "action": "clarify_deployment"
                }
        except Exception as e:
            return {"success": False, "error": f"Deployment error: {str(e)}"}
    
    def test_agent_from_conversation(self, user_input: str) -> Dict[str, Any]:
        """Test an agent based on conversational input."""
        try:
            return {
                "success": True,
                "message": "You can test your agents in the chat interface. Try asking them questions to see how they respond!",
                "action": "test_chat"
            }
        except Exception as e:
            return {"success": False, "error": f"Testing error: {str(e)}"}
    
    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents."""
        if agent_id:
            agent = self.agents.get(agent_id)
            if not agent:
                return {"success": False, "error": "Agent not found"}
            
            return {
                "success": True,
                "agent": agent,
                "status": "active",
                "knowledge_items": len(self.vector_manager.documents)
            }
        else:
            return {
                "success": True,
                "total_agents": len(self.agents),
                "total_knowledge_items": len(self.vector_manager.documents),
                "status": "system_healthy"
            }
    
    def upload_document(self, file_path: str, content: str = "") -> Dict[str, Any]:
        """Upload and process a document."""
        try:
            # Extract text from file
            if file_path.endswith('.pdf'):
                extracted_text = extract_text_from_pdf(file_path)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
                extracted_text = extract_text_from_image(file_path)
            else:
                return {"success": False, "error": "Unsupported file type"}
            
            # Combine with additional content
            full_content = f"{extracted_text}\n\n{content}".strip()
            
            # Add to vector store
            self.vector_manager.add_texts([full_content])
            
            return {
                "success": True,
                "message": f"Document processed and added to knowledge base",
                "content_length": len(full_content)
            }
        except Exception as e:
            return {"success": False, "error": f"Document processing failed: {str(e)}"}
    
    def train_from_url(self, url: str) -> Dict[str, Any]:
        """Train from a website URL."""
        try:
            content = extract_text_from_url(url)
            self.vector_manager.add_texts([content])
            
            return {
                "success": True,
                "message": f"Successfully trained from URL: {url}",
                "content_length": len(content)
            }
        except Exception as e:
            return {"success": False, "error": f"URL training failed: {str(e)}"}
    
    def add_knowledge(self, text: str) -> Dict[str, Any]:
        """Add knowledge text directly."""
        try:
            self.vector_manager.add_texts([text])
            
            return {
                "success": True,
                "message": "Knowledge added successfully",
                "content_length": len(text)
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to add knowledge: {str(e)}"}
    
    def generate_embed_code(self, agent_id: str) -> Dict[str, Any]:
        """Generate embed code for an agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"success": False, "error": "Agent not found"}
        
        # Generate embed code (simplified version)
        embed_code = f"""
<!-- AgentX Embed Code for {agent['name']} -->
<div id="agentx-widget"></div>
<script>
(function() {{
    const config = {json.dumps(agent)};
    const apiUrl = window.location.protocol + '//' + window.location.host;
    
    // Widget implementation here
    document.getElementById('agentx-widget').innerHTML = `
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
            <div style="background: #5b7cfa; color: white; padding: 15px; border-radius: 50%; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                💬
            </div>
        </div>
    `;
}})();
</script>
"""
        
        return {
            "success": True,
            "message": "Embed code generated successfully",
            "embed_code": embed_code
        }
    
    def get_share_url(self, agent_id: str) -> Dict[str, Any]:
        """Get shareable URL for an agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"success": False, "error": "Agent not found"}
        
        # In a real implementation, this would generate a proper shareable URL
        share_url = f"/agent/{agent_id}"
        
        return {
            "success": True,
            "message": "Share URL generated successfully",
            "share_url": share_url,
            "agent_name": agent['name']
        }
