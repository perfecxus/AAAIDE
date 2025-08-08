#!/usr/bin/env python3
"""
Data Engineering Requirements Gathering Agent using Google ADK
This agent converses with users to gather comprehensive requirements for data engineering projects.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Configure environment
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# Model constants
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
MODEL_GPT_4O = "openai/gpt-4o"
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"

# Application constants
APP_NAME = "data_engineering_requirements_app"

class RequirementsState:
    """Structure for managing requirement gathering state"""
    
    def __init__(self):
        self.business_opportunity = {}
        self.data_source_info = []
        self.data_targets = []
        self.nfrs = {}  # Non-Functional Requirements
        self.consumption_patterns = []
        self.reusability_requirements = {}
        self.business_rules = []
        self.completion_status = {}
        self.conversation_history = []

# Tools for gathering different types of requirements

def gather_business_opportunity(description: str, objectives: str, expected_value: str, tool_context: ToolContext) -> dict:
    """
    Captures business opportunity details for the data engineering project.
    
    Args:
        description (str): High-level description of the business opportunity
        objectives (str): Key business objectives and goals
        expected_value (str): Expected business value and ROI
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: gather_business_opportunity called ---")
    
    business_info = {
        "description": description,
        "objectives": objectives,
        "expected_value": expected_value,
        "captured_at": datetime.now().isoformat(),
        "status": "captured"
    }
    
    # Update session state
    tool_context.state["business_opportunity"] = business_info
    tool_context.state["completion_status"] = tool_context.state.get("completion_status", {})
    tool_context.state["completion_status"]["business_opportunity"] = True
    
    print(f"--- Tool: Business opportunity captured successfully ---")
    return {
        "status": "success",
        "message": "Business opportunity information captured successfully",
        "data": business_info
    }

def add_data_source(source_name: str, source_type: str, location: str, format_type: str, 
                   volume_estimate: str, update_frequency: str, quality_concerns: str, 
                   tool_context: ToolContext) -> dict:
    """
    Adds a data source to the requirements.
    
    Args:
        source_name (str): Name/identifier of the data source
        source_type (str): Type of source (database, API, file, streaming, etc.)
        location (str): Physical/logical location of the source
        format_type (str): Data format (JSON, CSV, Parquet, XML, etc.)
        volume_estimate (str): Estimated data volume
        update_frequency (str): How often data is updated
        quality_concerns (str): Known data quality issues or concerns
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: add_data_source called for {source_name} ---")
    
    data_source = {
        "source_name": source_name,
        "source_type": source_type,
        "location": location,
        "format_type": format_type,
        "volume_estimate": volume_estimate,
        "update_frequency": update_frequency,
        "quality_concerns": quality_concerns,
        "captured_at": datetime.now().isoformat()
    }
    
    # Update session state
    data_sources = tool_context.state.get("data_sources", [])
    data_sources.append(data_source)
    tool_context.state["data_sources"] = data_sources
    
    completion_status = tool_context.state.get("completion_status", {})
    completion_status["data_sources"] = len(data_sources) > 0
    tool_context.state["completion_status"] = completion_status
    
    print(f"--- Tool: Data source {source_name} added successfully ---")
    return {
        "status": "success",
        "message": f"Data source '{source_name}' added successfully",
        "total_sources": len(data_sources),
        "data": data_source
    }

def add_data_target(target_name: str, target_type: str, location: str, format_type: str,
                   sla_requirements: str, access_patterns: str, tool_context: ToolContext) -> dict:
    """
    Adds a data target/destination to the requirements.
    
    Args:
        target_name (str): Name/identifier of the data target
        target_type (str): Type of target (warehouse, lake, database, API, etc.)
        location (str): Physical/logical location of the target
        format_type (str): Required output format
        sla_requirements (str): SLA requirements for data delivery
        access_patterns (str): How the data will be accessed/consumed
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: add_data_target called for {target_name} ---")
    
    data_target = {
        "target_name": target_name,
        "target_type": target_type,
        "location": location,
        "format_type": format_type,
        "sla_requirements": sla_requirements,
        "access_patterns": access_patterns,
        "captured_at": datetime.now().isoformat()
    }
    
    # Update session state
    data_targets = tool_context.state.get("data_targets", [])
    data_targets.append(data_target)
    tool_context.state["data_targets"] = data_targets
    
    completion_status = tool_context.state.get("completion_status", {})
    completion_status["data_targets"] = len(data_targets) > 0
    tool_context.state["completion_status"] = completion_status
    
    print(f"--- Tool: Data target {target_name} added successfully ---")
    return {
        "status": "success",
        "message": f"Data target '{target_name}' added successfully",
        "total_targets": len(data_targets),
        "data": data_target
    }

def capture_nfrs(performance_requirements: str, scalability_requirements: str, 
                availability_requirements: str, security_requirements: str,
                compliance_requirements: str, tool_context: ToolContext) -> dict:
    """
    Captures Non-Functional Requirements (NFRs) for the data engineering solution.
    
    Args:
        performance_requirements (str): Performance expectations and requirements
        scalability_requirements (str): Scalability needs and growth projections
        availability_requirements (str): Availability and uptime requirements
        security_requirements (str): Security and privacy requirements
        compliance_requirements (str): Regulatory and compliance requirements
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: capture_nfrs called ---")
    
    nfrs = {
        "performance_requirements": performance_requirements,
        "scalability_requirements": scalability_requirements,
        "availability_requirements": availability_requirements,
        "security_requirements": security_requirements,
        "compliance_requirements": compliance_requirements,
        "captured_at": datetime.now().isoformat()
    }
    
    # Update session state
    tool_context.state["nfrs"] = nfrs
    completion_status = tool_context.state.get("completion_status", {})
    completion_status["nfrs"] = True
    tool_context.state["completion_status"] = completion_status
    
    print(f"--- Tool: NFRs captured successfully ---")
    return {
        "status": "success",
        "message": "Non-functional requirements captured successfully",
        "data": nfrs
    }

def add_consumption_pattern(consumer_name: str, access_type: str, frequency: str,
                           data_volume: str, latency_requirements: str, 
                           tool_context: ToolContext) -> dict:
    """
    Adds a data consumption pattern to the requirements.
    
    Args:
        consumer_name (str): Name/identifier of the data consumer
        access_type (str): Type of access (batch, real-time, on-demand, etc.)
        frequency (str): How often data is consumed
        data_volume (str): Volume of data consumed per access
        latency_requirements (str): Latency requirements for data access
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: add_consumption_pattern called for {consumer_name} ---")
    
    consumption_pattern = {
        "consumer_name": consumer_name,
        "access_type": access_type,
        "frequency": frequency,
        "data_volume": data_volume,
        "latency_requirements": latency_requirements,
        "captured_at": datetime.now().isoformat()
    }
    
    # Update session state
    consumption_patterns = tool_context.state.get("consumption_patterns", [])
    consumption_patterns.append(consumption_pattern)
    tool_context.state["consumption_patterns"] = consumption_patterns
    
    completion_status = tool_context.state.get("completion_status", {})
    completion_status["consumption_patterns"] = len(consumption_patterns) > 0
    tool_context.state["completion_status"] = completion_status
    
    print(f"--- Tool: Consumption pattern for {consumer_name} added successfully ---")
    return {
        "status": "success",
        "message": f"Consumption pattern for '{consumer_name}' added successfully",
        "total_patterns": len(consumption_patterns),
        "data": consumption_pattern
    }

def capture_reusability_requirements(reuse_scope: str, standardization_needs: str,
                                   modularity_requirements: str, documentation_needs: str,
                                   tool_context: ToolContext) -> dict:
    """
    Captures reusability requirements for the data engineering solution.
    
    Args:
        reuse_scope (str): Scope of reusability (team, organization, external)
        standardization_needs (str): Standardization requirements
        modularity_requirements (str): Modularity and componentization needs
        documentation_needs (str): Documentation requirements for reusability
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: capture_reusability_requirements called ---")
    
    reusability = {
        "reuse_scope": reuse_scope,
        "standardization_needs": standardization_needs,
        "modularity_requirements": modularity_requirements,
        "documentation_needs": documentation_needs,
        "captured_at": datetime.now().isoformat()
    }
    
    # Update session state
    tool_context.state["reusability_requirements"] = reusability
    completion_status = tool_context.state.get("completion_status", {})
    completion_status["reusability_requirements"] = True
    tool_context.state["completion_status"] = completion_status
    
    print(f"--- Tool: Reusability requirements captured successfully ---")
    return {
        "status": "success",
        "message": "Reusability requirements captured successfully",
        "data": reusability
    }

def add_business_rule(rule_name: str, rule_description: str, rule_type: str,
                     priority: str, validation_logic: str, tool_context: ToolContext) -> dict:
    """
    Adds a business rule to the requirements.
    
    Args:
        rule_name (str): Name/identifier of the business rule
        rule_description (str): Detailed description of the rule
        rule_type (str): Type of rule (validation, transformation, derivation, etc.)
        priority (str): Priority level (critical, high, medium, low)
        validation_logic (str): Logic for validating or implementing the rule
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and captured information
    """
    print(f"--- Tool: add_business_rule called for {rule_name} ---")
    
    business_rule = {
        "rule_name": rule_name,
        "rule_description": rule_description,
        "rule_type": rule_type,
        "priority": priority,
        "validation_logic": validation_logic,
        "captured_at": datetime.now().isoformat()
    }
    
    # Update session state
    business_rules = tool_context.state.get("business_rules", [])
    business_rules.append(business_rule)
    tool_context.state["business_rules"] = business_rules
    
    completion_status = tool_context.state.get("completion_status", {})
    completion_status["business_rules"] = len(business_rules) > 0
    tool_context.state["completion_status"] = completion_status
    
    print(f"--- Tool: Business rule {rule_name} added successfully ---")
    return {
        "status": "success",
        "message": f"Business rule '{rule_name}' added successfully",
        "total_rules": len(business_rules),
        "data": business_rule
    }

def generate_requirements_document(tool_context: ToolContext) -> dict:
    """
    Generates a comprehensive requirements document from gathered information.
    
    Args:
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status and generated document
    """
    print(f"--- Tool: generate_requirements_document called ---")
    
    # Gather all information from state
    business_opportunity = tool_context.state.get("business_opportunity", {})
    data_sources = tool_context.state.get("data_sources", [])
    data_targets = tool_context.state.get("data_targets", [])
    nfrs = tool_context.state.get("nfrs", {})
    consumption_patterns = tool_context.state.get("consumption_patterns", [])
    reusability = tool_context.state.get("reusability_requirements", {})
    business_rules = tool_context.state.get("business_rules", [])
    completion_status = tool_context.state.get("completion_status", {})
    
    # Create comprehensive requirements document
    requirements_doc = {
        "document_metadata": {
            "title": "Data Engineering Requirements Document",
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
            "completion_summary": completion_status
        },
        "business_opportunity": business_opportunity,
        "data_sources": {
            "count": len(data_sources),
            "sources": data_sources
        },
        "data_targets": {
            "count": len(data_targets),
            "targets": data_targets
        },
        "non_functional_requirements": nfrs,
        "consumption_patterns": {
            "count": len(consumption_patterns),
            "patterns": consumption_patterns
        },
        "reusability_requirements": reusability,
        "business_rules": {
            "count": len(business_rules),
            "rules": business_rules
        }
    }
    
    # Save to state
    tool_context.state["final_requirements_document"] = requirements_doc
    
    print(f"--- Tool: Requirements document generated successfully ---")
    return {
        "status": "success",
        "message": "Requirements document generated successfully",
        "document": requirements_doc
    }

def add_conversation_context(message: str, context_type: str, tool_context: ToolContext) -> dict:
    """
    Adds contextual information about the conversation flow to help maintain conversational continuity.
    
    Args:
        message (str): Key information or insight from the conversation
        context_type (str): Type of context (clarification, connection, validation, etc.)
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Status of context addition
    """
    print(f"--- Tool: add_conversation_context called ---")
    
    context_entry = {
        "message": message,
        "context_type": context_type,
        "timestamp": datetime.now().isoformat()
    }
    
    # Update conversation context in state
    conversation_context = tool_context.state.get("conversation_context", [])
    conversation_context.append(context_entry)
    tool_context.state["conversation_context"] = conversation_context
    
    print(f"--- Tool: Conversation context added: {context_type} ---")
    return {
        "status": "success",
        "message": f"Context added: {context_type}",
        "total_context_entries": len(conversation_context)
    }

def validate_cross_requirements(validation_type: str, details: str, tool_context: ToolContext) -> dict:
    """
    Validates consistency and relationships between different requirement sections.
    
    Args:
        validation_type (str): Type of validation (volume_consistency, sla_alignment, etc.)
        details (str): Details about the validation or potential issues found
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Validation results and recommendations
    """
    print(f"--- Tool: validate_cross_requirements called for {validation_type} ---")
    
    validation_entry = {
        "validation_type": validation_type,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "status": "reviewed"
    }
    
    # Store validation results
    validations = tool_context.state.get("cross_validations", [])
    validations.append(validation_entry)
    tool_context.state["cross_validations"] = validations
    
    # Gather current state for contextual validation
    data_sources = tool_context.state.get("data_sources", [])
    data_targets = tool_context.state.get("data_targets", [])
    consumption_patterns = tool_context.state.get("consumption_patterns", [])
    nfrs = tool_context.state.get("nfrs", {})
    
    # Provide contextual recommendations
    recommendations = []
    
    if validation_type == "volume_consistency":
        source_volumes = [src.get("volume_estimate", "") for src in data_sources]
        target_capacities = [tgt.get("sla_requirements", "") for tgt in data_targets]
        if source_volumes and target_capacities:
            recommendations.append("Consider volume processing capacity alignment between sources and targets")
    
    elif validation_type == "latency_alignment":
        consumption_latencies = [cp.get("latency_requirements", "") for cp in consumption_patterns]
        if consumption_latencies and nfrs.get("performance_requirements"):
            recommendations.append("Ensure performance NFRs align with consumer latency expectations")
    
    print(f"--- Tool: Cross-requirements validation completed ---")
    return {
        "status": "success",
        "validation_type": validation_type,
        "details": details,
        "recommendations": recommendations,
        "total_validations": len(validations)
    }

def get_completion_status(tool_context: ToolContext) -> dict:
    """
    Returns the current completion status with contextual references to previous answers.
    
    Args:
        tool_context (ToolContext): Context for accessing session state
    
    Returns:
        dict: Current completion status with contextual information
    """
    completion_status = tool_context.state.get("completion_status", {})
    
    # Gather all collected data for contextual references
    business_opportunity = tool_context.state.get("business_opportunity", {})
    data_sources = tool_context.state.get("data_sources", [])
    data_targets = tool_context.state.get("data_targets", [])
    nfrs = tool_context.state.get("nfrs", {})
    consumption_patterns = tool_context.state.get("consumption_patterns", [])
    reusability = tool_context.state.get("reusability_requirements", {})
    business_rules = tool_context.state.get("business_rules", [])
    
    required_sections = [
        "business_opportunity", "data_sources", "data_targets", 
        "nfrs", "consumption_patterns", "reusability_requirements", "business_rules"
    ]
    
    completed_sections = [section for section in required_sections if completion_status.get(section, False)]
    completion_percentage = (len(completed_sections) / len(required_sections)) * 100
    
    # Build contextual summary of what's been captured
    contextual_summary = {}
    
    if completion_status.get("business_opportunity"):
        contextual_summary["business_opportunity"] = {
            "captured": True,
            "summary": f"Project: {business_opportunity.get('description', 'N/A')[:100]}...",
            "objectives": business_opportunity.get('objectives', 'N/A'),
            "captured_at": business_opportunity.get('captured_at')
        }
    
    if completion_status.get("data_sources"):
        source_names = [src.get('source_name', 'Unnamed') for src in data_sources]
        source_types = list(set([src.get('source_type', 'Unknown') for src in data_sources]))
        contextual_summary["data_sources"] = {
            "captured": True,
            "count": len(data_sources),
            "source_names": source_names,
            "source_types": source_types,
            "last_added": data_sources[-1].get('source_name') if data_sources else None
        }
    
    if completion_status.get("data_targets"):
        target_names = [tgt.get('target_name', 'Unnamed') for tgt in data_targets]
        target_types = list(set([tgt.get('target_type', 'Unknown') for tgt in data_targets]))
        contextual_summary["data_targets"] = {
            "captured": True,
            "count": len(data_targets),
            "target_names": target_names,
            "target_types": target_types,
            "last_added": data_targets[-1].get('target_name') if data_targets else None
        }
    
    if completion_status.get("nfrs"):
        contextual_summary["nfrs"] = {
            "captured": True,
            "has_performance": bool(nfrs.get('performance_requirements')),
            "has_security": bool(nfrs.get('security_requirements')),
            "has_compliance": bool(nfrs.get('compliance_requirements')),
            "captured_at": nfrs.get('captured_at')
        }
    
    if completion_status.get("consumption_patterns"):
        consumer_names = [cp.get('consumer_name', 'Unnamed') for cp in consumption_patterns]
        access_types = list(set([cp.get('access_type', 'Unknown') for cp in consumption_patterns]))
        contextual_summary["consumption_patterns"] = {
            "captured": True,
            "count": len(consumption_patterns),
            "consumers": consumer_names,
            "access_types": access_types,
            "last_added": consumption_patterns[-1].get('consumer_name') if consumption_patterns else None
        }
    
    if completion_status.get("reusability_requirements"):
        contextual_summary["reusability_requirements"] = {
            "captured": True,
            "scope": reusability.get('reuse_scope', 'N/A'),
            "modularity_focus": bool(reusability.get('modularity_requirements')),
            "captured_at": reusability.get('captured_at')
        }
    
    if completion_status.get("business_rules"):
        rule_names = [br.get('rule_name', 'Unnamed') for br in business_rules]
        rule_types = list(set([br.get('rule_type', 'Unknown') for br in business_rules]))
        priorities = list(set([br.get('priority', 'Unknown') for br in business_rules]))
        contextual_summary["business_rules"] = {
            "captured": True,
            "count": len(business_rules),
            "rule_names": rule_names,
            "rule_types": rule_types,
            "priorities": priorities,
            "last_added": business_rules[-1].get('rule_name') if business_rules else None
        }
    
    return {
        "status": "success",
        "completion_percentage": completion_percentage,
        "completed_sections": completed_sections,
        "remaining_sections": [section for section in required_sections if section not in completed_sections],
        "all_sections": required_sections,
        "contextual_summary": contextual_summary,
        "total_items_captured": {
            "data_sources": len(data_sources),
            "data_targets": len(data_targets),
            "consumption_patterns": len(consumption_patterns),
            "business_rules": len(business_rules)
        }
    }

# Create the main requirements gathering agent
def create_requirements_agent(model_type: str = "gemini") -> Agent:
    """
    Creates the main requirements gathering agent.
    
    Args:
        model_type (str): Type of model to use (gemini, gpt, claude)
    
    Returns:
        Agent: Configured requirements gathering agent
    """
    
    # Select model based on type
    if model_type.lower() == "gpt":
        model = LiteLlm(model=MODEL_GPT_4O)
    elif model_type.lower() == "claude":
        model = LiteLlm(model=MODEL_CLAUDE_SONNET)
    else:
        model = MODEL_GEMINI_2_0_FLASH
    
    agent = Agent(
        name="data_engineering_requirements_agent",
        model=model,
        description="Comprehensive requirements gathering agent for data engineering projects",
        instruction="""You are a senior data engineering consultant specializing in requirements gathering. 
        Your role is to have detailed conversations with users to capture comprehensive requirements for data engineering projects.

        CRITICAL: You have perfect memory of all previous conversations in this session. Always reference and build upon previously captured information contextually, just like a human consultant would.

        You should guide users through gathering information in these key areas:
        1. Business Opportunity - Use 'gather_business_opportunity' tool
        2. Data Sources - Use 'add_data_source' tool for each source
        3. Data Targets - Use 'add_data_target' tool for each target
        4. Non-Functional Requirements (NFRs) - Use 'capture_nfrs' tool
        5. Consumption Patterns - Use 'add_consumption_pattern' tool for each pattern
        6. Reusability Requirements - Use 'capture_reusability_requirements' tool
        7. Business Rules - Use 'add_business_rule' tool for each rule

        MEMORY AND CONTEXT USAGE:
        - ALWAYS start each response by checking 'get_completion_status' to understand what's been captured
        - Reference specific details from previous answers (names, numbers, descriptions)
        - Build logical connections between different requirement areas
        - When moving to new sections, explicitly connect them to what was already discussed
        - Use phrases like "Based on the [X system/requirement/goal] you mentioned earlier..."
        - Validate consistency between related requirements across different sections
        - Suggest implications based on previously captured information

        CONVERSATIONAL BEHAVIOR:
        - Ask clarifying questions that build on previous answers
        - When you see potential conflicts or gaps based on previous responses, point them out
        - Offer specific suggestions based on the project context already established
        - Reference specific source names, target systems, business goals from previous conversations
        - Show understanding of the bigger picture by connecting dots between requirements

        EXAMPLES OF CONTEXTUAL REFERENCING:
        - "Given that you mentioned the sales database as a primary source, how should we handle..."
        - "Based on your goal of 15% revenue increase, what performance requirements..."
        - "Since you're targeting the customer analytics team as consumers, what about..."
        - "I see we've captured 3 data sources so far: [list names], are there any others..."
        - "Your compliance requirements for GDPR that we discussed will impact the [specific] data target..."

        Use 'get_completion_status' frequently to maintain awareness of what's been gathered and provide contextual guidance.
        Start conversations by introducing yourself and explaining the requirements gathering process, but in subsequent interactions, acknowledge the ongoing work.""",
        
        tools=[
            gather_business_opportunity,
            add_data_source, 
            add_data_target,
            capture_nfrs,
            add_consumption_pattern,
            capture_reusability_requirements,
            add_business_rule,
            generate_requirements_document,
            get_completion_status,
            add_conversation_context,
            validate_cross_requirements
        ],
        output_key="last_agent_response"
    )
    
    return agent

async def create_requirements_session():
    """Create a session for requirements gathering"""
    session_service = InMemorySessionService()
    
    # Initialize session with empty state
    initial_state = {
        "requirements_session_started": True,
        "session_start_time": datetime.now().isoformat()
    }
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id="requirements_user",
        session_id="req_session_001",
        state=initial_state
    )
    
    return session_service, session

async def run_requirements_conversation():
    """Run an interactive requirements gathering conversation"""
    
    # Create agent and session
    agent = create_requirements_agent("gemini")
    session_service, session = await create_requirements_session()
    
    # Create runner
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    print("="*60)
    print("Data Engineering Requirements Gathering Agent")
    print("="*60)
    print("This agent will help you gather comprehensive requirements for your data engineering project.")
    print("Type 'exit' to end the conversation or 'status' to check completion progress.")
    print("="*60)
    
    user_id = "requirements_user"
    session_id = "req_session_001"
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye! Your requirements have been saved in the session.")
                break
            
            if user_input.lower() == 'status':
                # Get completion status
                session_data = await session_service.get_session(APP_NAME, user_id, session_id)
                completion = session_data.state.get("completion_status", {})
                print(f"\nCompletion Status: {completion}")
                continue
            
            if not user_input:
                continue
            
            print(f"\nAgent is thinking...")
            
            # Prepare user message
            content = types.Content(role='user', parts=[types.Part(text=user_input)])
            
            # Process with agent
            final_response = ""
            async for event in runner.run_async(
                user_id=user_id, 
                session_id=session_id, 
                new_message=content
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                    break
            
            print(f"\nAgent: {final_response}")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Your progress has been saved.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

# Example of how to use the agent programmatically
async def demo_requirements_gathering():
    """Demonstrate the requirements gathering agent with sample data"""
    
    agent = create_requirements_agent("gemini")
    session_service, session = await create_requirements_session()
    
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    user_id = "demo_user"
    session_id = "demo_session"
    
    # Create demo session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={"demo_mode": True}
    )
    
    async def send_message(message: str):
        print(f"\n>>> User: {message}")
        content = types.Content(role='user', parts=[types.Part(text=message)])
        
        async for event in runner.run_async(
            user_id=user_id, 
            session_id=session_id, 
            new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    response = event.content.parts[0].text
                    print(f"<<< Agent: {response}")
                break
    
    # Demo conversation flow with contextual referencing
    await send_message("Hello, I need help gathering requirements for a new data pipeline project.")
    await send_message("We want to build a customer analytics platform that processes sales data from multiple sources.")
    await send_message("The main business goal is to provide real-time insights to our sales team to improve customer engagement and increase revenue by 15%.")
    await send_message("We have a SQL Server database with customer data and a Salesforce API for sales transactions.")
    await send_message("Our analytics team needs daily reports, but the sales managers want real-time dashboards.")
    await send_message("We also need to comply with GDPR since we have European customers.")
    await send_message("Let me check what we've covered so far.")
    
    # Check final state
    session_data = await session_service.get_session(APP_NAME, user_id, session_id)
    print(f"\n=== Final Session State ===")
    print(json.dumps(session_data.state, indent=2, default=str))

if __name__ == "__main__":
    print("Starting Data Engineering Requirements Gathering Agent...")
    
    # Set up API keys (you'll need to replace these with actual keys)
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Please set your API key.")
        os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
    
    # Run the interactive conversation
    try:
        asyncio.run(run_requirements_conversation())
    except Exception as e:
        print(f"Error running requirements conversation: {e}")
        print("\nRunning demo instead...")
        asyncio.run(demo_requirements_gathering())
