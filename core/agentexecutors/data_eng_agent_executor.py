#!/usr/bin/env python3
"""
Agent Executor for Data Engineering Requirements Gathering Agent
Compatible with Google's A2A SDK specifications
"""

import os
import json
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Import the main agent components (assuming they're in the same module or imported)
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    ERROR = "error"


class MessageType(Enum):
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_MESSAGE = "system_message"
    TOOL_EXECUTION = "tool_execution"
    STATUS_UPDATE = "status_update"


@dataclass
class AgentMessage:
    """Structured message format for A2A communication"""
    message_id: str
    session_id: str
    timestamp: str
    message_type: MessageType
    content: str
    metadata: Dict[str, Any]
    agent_id: str = "data-engineering-requirements-agent-v1"


@dataclass
class AgentSession:
    """Agent session management"""
    session_id: str
    user_id: str
    status: AgentStatus
    created_at: str
    last_activity: str
    completion_percentage: float
    context_summary: Dict[str, Any]
    message_history: List[AgentMessage]


class DataEngineeringRequirementsAgentExecutor:
    """
    Agent Executor implementing Google's A2A SDK patterns for the 
    Data Engineering Requirements Gathering Agent
    """
    
    def __init__(self, 
                 model_type: str = "gemini",
                 app_name: str = "data_engineering_requirements_app"):
        """
        Initialize the Agent Executor
        
        Args:
            model_type: Type of model to use (gemini, gpt, claude)
            app_name: Application name for session management
        """
        self.agent_id = "data-engineering-requirements-agent-v1"
        self.app_name = app_name
        self.model_type = model_type
        
        # Initialize agent components
        self.agent = None
        self.session_service = None
        self.runner = None
        
        # Session management
        self.active_sessions: Dict[str, AgentSession] = {}
        
        # Agent card metadata
        self.agent_card = self._load_agent_card()
        
        # Initialize the agent
        self._initialize_agent()
    
    def _load_agent_card(self) -> Dict[str, Any]:
        """Load agent card metadata"""
        # In a real implementation, this would load from the JSON file
        return {
            "agent_id": self.agent_id,
            "agent_name": "Data Engineering Requirements Gathering Agent",
            "version": "1.0.0",
            "capabilities": [
                "Business opportunity analysis",
                "Data source specification",
                "Technical requirements capture",
                "Contextual conversation management"
            ]
        }
    
    def _initialize_agent(self):
        """Initialize the core agent components"""
        try:
            # Import and create the agent (assuming the agent creation function exists)
            from data_eng_requirements_agent import create_requirements_agent
            
            self.agent = create_requirements_agent(self.model_type)
            self.session_service = InMemorySessionService()
            
            print(f"✅ Agent Executor initialized with model: {self.model_type}")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            raise
    
    async def create_session(self, user_id: str, 
                           initial_context: Optional[Dict[str, Any]] = None) -> AgentSession:
        """
        Create a new agent session
        
        Args:
            user_id: Unique identifier for the user
            initial_context: Optional initial context for the session
            
        Returns:
            AgentSession: Created session object
        """
        session_id = f"req_session_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        
        # Initialize session state
        initial_state = {
            "requirements_session_started": True,
            "session_start_time": timestamp,
            "user_id": user_id,
            "agent_executor_version": "1.0.0"
        }
        
        if initial_context:
            initial_state.update(initial_context)
        
        # Create ADK session
        adk_session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            state=initial_state
        )
        
        # Create runner for this session
        runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )
        
        # Create agent session
        agent_session = AgentSession(
            session_id=session_id,
            user_id=user_id,
            status=AgentStatus.IDLE,
            created_at=timestamp,
            last_activity=timestamp,
            completion_percentage=0.0,
            context_summary={},
            message_history=[]
        )
        
        self.active_sessions[session_id] = agent_session
        
        print(f"✅ Created session: {session_id} for user: {user_id}")
        return agent_session
    
    async def send_message(self, session_id: str, message: str, 
                          message_type: MessageType = MessageType.USER_INPUT,
                          metadata: Optional[Dict[str, Any]] = None) -> AgentMessage:
        """
        Send a message to the agent and get response
        
        Args:
            session_id: Session identifier
            message: Message content
            message_type: Type of message
            metadata: Additional metadata
            
        Returns:
            AgentMessage: Agent's response
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session.status = AgentStatus.PROCESSING
        session.last_activity = datetime.now().isoformat()
        
        # Create input message
        input_message = AgentMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            message_type=message_type,
            content=message,
            metadata=metadata or {},
            agent_id=self.agent_id
        )
        
        session.message_history.append(input_message)
        
        try:
            # Create runner for this session if not exists
            runner = Runner(
                agent=self.agent,
                app_name=self.app_name,
                session_service=self.session_service
            )
            
            # Prepare message for ADK
            content = types.Content(role='user', parts=[types.Part(text=message)])
            
            # Process with agent
            agent_response = ""
            async for event in runner.run_async(
                user_id=session.user_id,
                session_id=session_id,
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        agent_response = event.content.parts[0].text
                    break
            
            # Update session status
            await self._update_session_status(session_id)
            
            # Create response message
            response_message = AgentMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                message_type=MessageType.AGENT_RESPONSE,
                content=agent_response,
                metadata={
                    "processing_time": "calculated_processing_time",
                    "tools_used": "extracted_from_event_history"
                },
                agent_id=self.agent_id
            )
            
            session.message_history.append(response_message)
            session.status = AgentStatus.WAITING_FOR_INPUT
            
            return response_message
            
        except Exception as e:
            session.status = AgentStatus.ERROR
            error_message = AgentMessage(
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                message_type=MessageType.SYSTEM_MESSAGE,
                content=f"Error processing message: {str(e)}",
                metadata={"error_type": "processing_error"},
                agent_id=self.agent_id
            )
            session.message_history.append(error_message)
            raise
    
    async def _update_session_status(self, session_id: str):
        """Update session status and completion percentage"""
        session = self.active_sessions[session_id]
        
        # Get current state from ADK session
        adk_session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=session.user_id,
            session_id=session_id
        )
        
        if adk_session:
            completion_status = adk_session.state.get("completion_status", {})
            required_sections = [
                "business_opportunity", "data_sources", "data_targets",
                "nfrs", "consumption_patterns", "reusability_requirements", 
                "business_rules"
            ]
            
            completed = sum(1 for section in required_sections 
                          if completion_status.get(section, False))
            session.completion_percentage = (completed / len(required_sections)) * 100
            
            # Update context summary
            session.context_summary = {
                "completed_sections": completed,
                "total_sections": len(required_sections),
                "business_opportunity": adk_session.state.get("business_opportunity", {}),
                "data_sources_count": len(adk_session.state.get("data_sources", [])),
                "data_targets_count": len(adk_session.state.get("data_targets", [])),
                "last_updated": datetime.now().isoformat()
            }
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session status
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict: Session status information
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        await self._update_session_status(session_id)
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "completion_percentage": session.completion_percentage,
            "context_summary": session.context_summary,
            "message_count": len(session.message_history),
            "last_activity": session.last_activity,
            "agent_info": self.agent_card
        }
    
    async def generate_requirements_document(self, session_id: str) -> Dict[str, Any]:
        """
        Generate final requirements document
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict: Generated requirements document
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Send document generation request to agent
        response = await self.send_message(
            session_id=session_id,
            message="Please generate the final requirements document with all the information we've gathered.",
            message_type=MessageType.SYSTEM_MESSAGE
        )
        
        # Get the document from session state
        adk_session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=self.active_sessions[session_id].user_id,
            session_id=session_id
        )
        
        final_document = adk_session.state.get("final_requirements_document", {})
        
        return {
            "document": final_document,
            "generation_response": response.content,
            "generated_at": datetime.now().isoformat(),
            "session_id": session_id
        }
    
    async def export_session(self, session_id: str, 
                           format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export session data in specified format
        
        Args:
            session_id: Session identifier
            format_type: Export format (json, yaml, etc.)
            
        Returns:
            Exported session data
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Get full session state
        adk_session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=session.user_id,
            session_id=session_id
        )
        
        export_data = {
            "session_metadata": asdict(session),
            "requirements_data": adk_session.state if adk_session else {},
            "agent_card": self.agent_card,
            "export_timestamp": datetime.now().isoformat()
        }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities from the agent card"""
        return {
            "agent_id": self.agent_id,
            "capabilities": self.agent_card.get("capabilities", []),
            "supported_operations": [
                "create_session",
                "send_message", 
                "get_session_status",
                "generate_requirements_document",
                "export_session"
            ],
            "tools_available": [
                "gather_business_opportunity",
                "add_data_source",
                "add_data_target", 
                "capture_nfrs",
                "add_consumption_pattern",
                "capture_reusability_requirements",
                "add_business_rule",
                "get_completion_status",
                "validate_cross_requirements",
                "generate_requirements_document"
            ]
        }


# API-style interface for integration
class RequirementsAgentAPI:
    """
    API wrapper for the Agent Executor
    Provides REST-like interface for external integrations
    """
    
    def __init__(self, model_type: str = "gemini"):
        self.executor = DataEngineeringRequirementsAgentExecutor(model_type=model_type)
    
    async def create_session_endpoint(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """POST /requirements/session/create"""
        user_id = request.get("user_id", f"user_{uuid.uuid4().hex[:8]}")
        initial_context = request.get("initial_context", {})
        
        session = await self.executor.create_session(user_id, initial_context)
        
        return {
            "success": True,
            "session_id": session.session_id,
            "status": session.status.value,
            "message": "Session created successfully"
        }
    
    async def send_message_endpoint(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """POST /requirements/session/message"""
        session_id = request["session_id"]
        message = request["message"]
        
        response = await self.executor.send_message(session_id, message)
        
        return {
            "success": True,
            "response": response.content,
            "message_id": response.message_id,
            "timestamp": response.timestamp
        }
    
    async def get_status_endpoint(self, session_id: str) -> Dict[str, Any]:
        """GET /requirements/session/{session_id}/status"""
        status = await self.executor.get_session_status(session_id)
        
        return {
            "success": True,
            "status": status
        }
    
    async def generate_document_endpoint(self, session_id: str) -> Dict[str, Any]:
        """POST /requirements/session/{session_id}/document"""
        document = await self.executor.generate_requirements_document(session_id)
        
        return {
            "success": True,
            "document": document
        }


# Example usage and testing
async def demo_agent_executor():
    """Demonstrate the Agent Executor functionality"""
    
    print("=== Data Engineering Requirements Agent Executor Demo ===")
    
    # Initialize executor
    executor = DataEngineeringRequirementsAgentExecutor(model_type="gemini")
    
    # Create session
    session = await executor.create_session("demo_user", {"demo_mode": True})
    print(f"Created session: {session.session_id}")
    
    # Send messages
    messages = [
        "Hello, I need help with requirements for a customer analytics project.",
        "We want to analyze customer behavior to increase sales by 20%.",
        "Our main data sources are a PostgreSQL customer database and Stripe payment API.",
        "What's our current progress?"
    ]
    
    for msg in messages:
        print(f"\n>>> Sending: {msg}")
        response = await executor.send_message(session.session_id, msg)
        print(f"<<< Response: {response.content[:200]}...")
    
    # Get status
    status = await executor.get_session_status(session.session_id)
    print(f"\n=== Session Status ===")
    print(f"Completion: {status['completion_percentage']}%")
    print(f"Status: {status['status']}")
    
    # Export session
    export_data = await executor.export_session(session.session_id)
    print(f"\n=== Export completed, length: {len(export_data)} characters ===")


if __name__ == "__main__":
    # Set up environment
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: Please set GOOGLE_API_KEY environment variable")
        os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
    
    # Run demo
    asyncio.run(demo_agent_executor())
