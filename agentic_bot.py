import os
from typing import List, Tuple

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from train_data import VectorStoreManager
from emotion_detector import detect_user_emotion, get_emotion_context_for_prompt


RAG_PROMPT = PromptTemplate.from_template(
    """
You are an expert assistant. Use the provided context to answer the question accurately and concisely.
If the answer is not in the context, say you don't know. Do not hallucinate.

Context:
{context}

Chat History:
{history}

Question: {question}

Helpful answer:
"""
)


class AgenticBot:
    """Lightweight Retrieval-Augmented-Generation wrapper around Ollama and Chroma."""

    def __init__(self, vector_manager: VectorStoreManager, model: str = "llama3.2:3b") -> None:
        self.vector_manager = vector_manager
        self.llm = Ollama(model=model)

    def _format_history(self, history: List[Tuple[str, str]]) -> str:
        # Format history as alternating User/Assistant lines to help the model
        lines = []
        for q, a in history[-8:]:  # last 8 exchanges to keep prompt compact
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def answer_question(self, question: str, chat_history: List[Tuple[str, str]] = None, agent_id: str = None):
        # Detect user emotion
        emotion_data = detect_user_emotion(question)
        emotion_context = get_emotion_context_for_prompt(question)
        
        # Use agent-specific vector store if agent_id is provided
        if agent_id:
            try:
                # Import agent_tools to get agent vector store path
                from agent_tools import AgentTools
                agent_tools = AgentTools("agents", "vector_store")
                
                if agent_id not in agent_tools.agents:
                    return f"I'm sorry, I couldn't find the agent with ID {agent_id}.", [], emotion_data
                
                agent = agent_tools.agents[agent_id]
                vector_store_path = agent.get("vector_store_path")
                
                if not vector_store_path or not os.path.exists(vector_store_path):
                    return f"I'm sorry, but I haven't been trained with any knowledge yet. Please train me with some documents or URLs first.", [], emotion_data
                
                # Create agent-specific vector manager
                from train_data import VectorStoreManager
                agent_vector_manager = VectorStoreManager(persist_directory=vector_store_path)
                
                # Retrieve documents from agent-specific vector store
                docs = agent_vector_manager.similarity_search(question, k=5)
            except Exception as e:
                return f"I encountered an error accessing my knowledge base: {str(e)}", [], emotion_data
        else:
            # Use global vector store for general queries
            docs = self.vector_manager.similarity_search(question, k=5)
        
        # Extract context from documents
        context_parts = []
        for doc in docs:
            if isinstance(doc, dict):
                context_parts.append(doc.get("page_content", ""))
            else:
                context_parts.append(getattr(doc, "page_content", ""))
        
        context = "\n\n".join(context_parts)
        history_str = self._format_history(chat_history or [])

        # Build enhanced prompt with emotion context
        enhanced_prompt = RAG_PROMPT.format(context=context, history=history_str, question=question)
        if emotion_context:
            enhanced_prompt = f"{emotion_context}\n\n{enhanced_prompt}"
        
        answer = self.llm.invoke(enhanced_prompt)

        # Extract sources metadata
        sources = []
        for doc in docs:
            if isinstance(doc, dict):
                sources.append(doc.get("metadata", {}))
            else:
                sources.append(getattr(doc, "metadata", {}))
        
        return answer, sources, emotion_data


