"""
Conversational AI system for AgentX-style interface.
Handles natural language interaction and tool calling.
"""

import json
import re
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate


class ConversationalAI:
    """Main conversational AI system that handles user interactions and tool calling."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.llm = Ollama(model=model, temperature=0.7)
        self.system_prompt = self._get_system_prompt()
        self.conversation_history: List[Dict[str, str]] = []
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the conversational AI."""
        return """You are AgentX, an advanced AI assistant specialized in creating, managing, and deploying AI agents and chatbots. You have access to powerful tools to help users build sophisticated AI systems through natural conversation.

Your capabilities include:
- Creating and configuring AI agents with custom personalities, greetings, and behaviors
- Training agents on documents, websites, and custom knowledge bases
- Managing agent settings, names, descriptions, and deployment options
- Generating embed codes for websites
- Testing and debugging agent responses
- Providing guidance on AI agent best practices

When users ask you to:
1. Create an agent: Use the create_agent tool with name, greeting, and description
2. Train an agent: Use training tools (upload_document, train_from_url, add_knowledge)
3. Deploy an agent: Use deployment tools (generate_embed_code, get_share_url)
4. Test an agent: Use the test_agent tool
5. Manage agents: Use list_agents, update_agent, delete_agent tools

Always be helpful, professional, and proactive. When users describe what they want, suggest the best approach and execute it using your tools.

Current time: {current_time}

Available tools: create_agent, update_agent, list_agents, delete_agent, upload_document, train_from_url, add_knowledge, test_agent, generate_embed_code, get_share_url, clear_session, get_agent_status

Remember: You can perform actions directly through tools. Don't just describe what you would do - actually do it when appropriate."""

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_conversation_context(self) -> str:
        """Get formatted conversation history for context."""
        context = ""
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            role = msg["role"]
            content = msg["content"]
            context += f"{role}: {content}\n"
        return context

    def process_user_input(self, user_input: str, tools: 'AgentTools') -> Dict[str, Any]:
        """Process user input and generate response with potential tool calls."""
        # Add user input to history
        self.add_to_history("user", user_input)
        
        # Check if user input contains tool requests
        tool_calls = self._detect_tool_requests(user_input)
        
        # If no specific tool requests, use conversational AI
        if not tool_calls:
            response = self._generate_conversational_response(user_input)
            self.add_to_history("assistant", response)
            return {
                "type": "conversation",
                "content": response,
                "tool_calls": []
            }
        
        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            result = self._execute_tool_call(tool_call, tools)
            tool_results.append(result)
        
        # Generate response based on tool results
        response = self._generate_tool_response(user_input, tool_results)
        self.add_to_history("assistant", response)
        
        return {
            "type": "tool_execution",
            "content": response,
            "tool_calls": tool_results
        }

    def _detect_tool_requests(self, user_input: str) -> List[Dict[str, Any]]:
        """Detect if user input contains requests that should trigger tools."""
        user_input_lower = user_input.lower()
        tool_calls = []
        
        # Pattern matching for common requests
        patterns = {
            "create_agent": [
                r"create.*agent", r"make.*bot", r"build.*assistant", 
                r"new.*agent", r"start.*agent"
            ],
            "list_agents": [
                r"list.*agent", r"show.*agent", r"what.*agent", 
                r"my.*agent", r"all.*agent"
            ],
            "train_agent": [
                r"train.*agent", r"upload.*document", r"add.*knowledge",
                r"teach.*agent", r"feed.*data"
            ],
            "deploy_agent": [
                r"deploy.*agent", r"embed.*code", r"share.*agent",
                r"website.*embed", r"public.*share"
            ],
            "test_agent": [
                r"test.*agent", r"try.*agent", r"chat.*with.*agent"
            ]
        }
        
        for tool_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, user_input_lower):
                    tool_calls.append({
                        "tool": tool_name,
                        "input": user_input,
                        "confidence": 0.8
                    })
                    break
        
        return tool_calls

    def _execute_tool_call(self, tool_call: Dict[str, Any], tools: 'AgentTools') -> Dict[str, Any]:
        """Execute a tool call."""
        tool_name = tool_call["tool"]
        user_input = tool_call["input"]
        
        try:
            if tool_name == "create_agent":
                return tools.create_agent_from_conversation(user_input)
            elif tool_name == "list_agents":
                return tools.list_agents()
            elif tool_name == "train_agent":
                return tools.train_agent_from_conversation(user_input)
            elif tool_name == "deploy_agent":
                return tools.deploy_agent_from_conversation(user_input)
            elif tool_name == "test_agent":
                return tools.test_agent_from_conversation(user_input)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_conversational_response(self, user_input: str) -> str:
        """Generate a conversational response without tool calls."""
        context = self.get_conversation_context()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""{self.system_prompt.format(current_time=current_time)}

Conversation History:
{context}

User: {user_input}

AgentX:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Error: {str(e)}"

    def _generate_tool_response(self, user_input: str, tool_results: List[Dict[str, Any]]) -> str:
        """Generate response based on tool execution results."""
        if not tool_results:
            return "I understand your request, but I couldn't execute any specific actions."
        
        successful_results = [r for r in tool_results if r.get("success", False)]
        failed_results = [r for r in tool_results if not r.get("success", False)]
        
        response_parts = []
        
        if successful_results:
            response_parts.append("I've successfully completed the following actions:")
            for result in successful_results:
                response_parts.append(f"✅ {result.get('message', 'Action completed')}")
        
        if failed_results:
            response_parts.append("\nHowever, I encountered some issues:")
            for result in failed_results:
                response_parts.append(f"❌ {result.get('error', 'Unknown error')}")
        
        return "\n".join(response_parts)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()
