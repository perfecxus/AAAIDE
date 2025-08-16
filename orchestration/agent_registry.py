#!/usr/bin/env python3
"""
A2A Agent Registry Backend

This module implements the Agent Registry service for the A2A protocol
as specified in the GitHub proposal: https://github.com/a2aproject/A2A/discussions/741

Usage:
    pip install fastapi uvicorn httpx pydantic
    python agent_registry.py

API Endpoints:
    GET /agents/public - Public agent discovery (no auth required)
    GET /agents/entitled - Entitled agents (requires Bearer token)
    POST /agents/search - Search agents (requires Bearer token)
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import json
import uuid
from datetime import datetime
import sqlite3
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class AgentLocation(BaseModel):
    url: str
    type: str = "agent_card"


class AgentResource(BaseModel):
    id: str
    name: str
    location: AgentLocation
    description: Optional[str] = None
    tags: List[str] = []
    is_public: bool = False
    provider_organization: Optional[str] = None
    added_at: Optional[datetime] = None


class RegistryResponse(BaseModel):
    registry_version: str = "0.1"
    resources: List[AgentResource]
    total_count: int
    offset: Optional[int] = None
    limit: Optional[int] = None


class SearchRequest(BaseModel):
    query: str
    capabilities: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    provider: Optional[str] = None


class ClientRegistration(BaseModel):
    name: str
    description: Optional[str] = None
    redirect_uris: Optional[List[str]] = None


class ClientResponse(BaseModel):
    client_id: str
    client_secret: str
    name: str
    description: Optional[str]
    created_at: datetime


class EntitlementRequest(BaseModel):
    client_id: str
    agent_ids: List[str]


# Database setup
class DatabaseManager:
    def __init__(self, db_path: str = "agent_registry.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Agents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                agent_card_url TEXT NOT NULL,
                tags TEXT,
                is_public BOOLEAN DEFAULT FALSE,
                provider_organization TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                added_by TEXT,
                agent_card_content TEXT
            )
        ''')

        # Clients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                client_secret TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        ''')

        # Entitlements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entitlements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                granted_by TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (client_id),
                FOREIGN KEY (agent_id) REFERENCES agents (id),
                UNIQUE(client_id, agent_id)
            )
        ''')

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                role TEXT CHECK(role IN ('Administrator', 'Catalog Manager', 'User', 'Viewer')) DEFAULT 'Viewer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def get_connection(self):
        return sqlite3.connect(self.db_path)


# Initialize database
db_manager = DatabaseManager()

# Security
security = HTTPBearer()


class AuthenticationService:
    """Simple authentication service for demonstration"""

    def __init__(self):
        # In production, this would integrate with OAuth2/OIDC provider
        self.valid_tokens = {
            "demo-client-token": {
                "client_id": "demo-client",
                "scopes": ["read", "search"]
            },
            "admin-token": {
                "client_id": "admin-client",
                "scopes": ["read", "write", "admin"]
            }
        }

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Bearer token and return client info"""
        return self.valid_tokens.get(token)


auth_service = AuthenticationService()


async def get_current_client(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated client"""
    token = credentials.credentials
    client_info = auth_service.verify_token(token)

    if not client_info:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    return client_info


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Seed database with sample data
    await seed_sample_data()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="A2A Agent Registry",
    description="Agent discovery and entitlement service for A2A protocol",
    version="0.1.0",
    lifespan=lifespan
)


async def seed_sample_data():
    """Seed database with sample agents and clients for demonstration"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    # Sample agents
    sample_agents = [
        {
            "id": "georoute-agent",
            "name": "GeoSpatial Route Planner Agent",
            "description": "Provides advanced route planning, traffic analysis, and custom map generation services",
            "agent_card_url": "https://georoute-agent.example.com/.well-known/agent.json",
            "tags": "maps,routing,navigation,traffic",
            "is_public": True,
            "provider_organization": "Example Geo Services Inc."
        },
        {
            "id": "benefits-agent",
            "name": "Benefits Agent",
            "description": "HR benefits information and policy agent",
            "agent_card_url": "https://benefits.example.com/.well-known/agent.json",
            "tags": "hr,benefits,policy,401k",
            "is_public": False,
            "provider_organization": "HR Solutions Corp"
        },
        {
            "id": "it-agent",
            "name": "IT Support Agent",
            "description": "IT helpdesk and support services agent",
            "agent_card_url": "https://it.example.com/.well-known/agent.json",
            "tags": "it,support,helpdesk,tickets",
            "is_public": False,
            "provider_organization": "TechSupport Inc."
        }
    ]

    for agent in sample_agents:
        cursor.execute('''
            INSERT OR REPLACE INTO agents 
            (id, name, description, agent_card_url, tags, is_public, provider_organization)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            agent["id"], agent["name"], agent["description"],
            agent["agent_card_url"], agent["tags"], agent["is_public"],
            agent["provider_organization"]
        ))

    # Sample clients
    cursor.execute('''
        INSERT OR REPLACE INTO clients (client_id, client_secret, name, description)
        VALUES (?, ?, ?, ?)
    ''', ("demo-client", "demo-secret", "Demo Application", "Sample client application"))

    # Sample entitlements
    cursor.execute('''
        INSERT OR REPLACE INTO entitlements (client_id, agent_id)
        VALUES (?, ?)
    ''', ("demo-client", "benefits-agent"))

    cursor.execute('''
        INSERT OR REPLACE INTO entitlements (client_id, agent_id)
        VALUES (?, ?)
    ''', ("demo-client", "it-agent"))

    conn.commit()
    conn.close()
    logger.info("Sample data seeded successfully")


class AgentRegistryService:
    """Core service for agent registry operations"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def fetch_agent_card(self, agent_card_url: str) -> Optional[Dict[str, Any]]:
        """Fetch and cache agent card from URL"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(agent_card_url, timeout=10.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch agent card from {agent_card_url}: {e}")
            return None

    def get_public_agents(self, offset: int = 0, limit: int = 100) -> RegistryResponse:
        """Get all public agents"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM agents WHERE is_public = TRUE")
        total_count = cursor.fetchone()[0]

        # Get agents with pagination
        cursor.execute('''
            SELECT id, name, description, agent_card_url, tags, provider_organization, added_at
            FROM agents 
            WHERE is_public = TRUE
            ORDER BY added_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))

        agents = []
        for row in cursor.fetchall():
            agent_id, name, description, url, tags_str, provider, added_at = row
            tags = tags_str.split(',') if tags_str else []

            agents.append(AgentResource(
                id=agent_id,
                name=name,
                description=description,
                location=AgentLocation(url=url),
                tags=tags,
                is_public=True,
                provider_organization=provider,
                added_at=datetime.fromisoformat(added_at) if added_at else None
            ))

        conn.close()

        return RegistryResponse(
            resources=agents,
            total_count=total_count,
            offset=offset,
            limit=limit
        )

    def get_entitled_agents(self, client_id: str, offset: int = 0, limit: int = 100) -> RegistryResponse:
        """Get agents entitled to a specific client, including public agents"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Get entitled + public agents
        query = '''
            SELECT DISTINCT a.id, a.name, a.description, a.agent_card_url, a.tags, 
                   a.provider_organization, a.added_at, a.is_public
            FROM agents a
            LEFT JOIN entitlements e ON a.id = e.agent_id
            WHERE a.is_public = TRUE OR e.client_id = ?
            ORDER BY a.added_at DESC
            LIMIT ? OFFSET ?
        '''

        cursor.execute(query, (client_id, limit, offset))

        agents = []
        for row in cursor.fetchall():
            agent_id, name, description, url, tags_str, provider, added_at, is_public = row
            tags = tags_str.split(',') if tags_str else []

            agents.append(AgentResource(
                id=agent_id,
                name=name,
                description=description,
                location=AgentLocation(url=url),
                tags=tags,
                is_public=bool(is_public),
                provider_organization=provider,
                added_at=datetime.fromisoformat(added_at) if added_at else None
            ))

        # Get total count
        count_query = '''
            SELECT COUNT(DISTINCT a.id)
            FROM agents a
            LEFT JOIN entitlements e ON a.id = e.agent_id
            WHERE a.is_public = TRUE OR e.client_id = ?
        '''
        cursor.execute(count_query, (client_id,))
        total_count = cursor.fetchone()[0]

        conn.close()

        return RegistryResponse(
            resources=agents,
            total_count=total_count,
            offset=offset,
            limit=limit
        )

    def search_agents(self, client_id: str, search_request: SearchRequest,
                      offset: int = 0, limit: int = 100) -> RegistryResponse:
        """Search agents based on query and filters"""
        conn = self.db_manager.get_connection()
        cursor = conn.cursor()

        # Build search query
        base_query = '''
            SELECT DISTINCT a.id, a.name, a.description, a.agent_card_url, a.tags,
                   a.provider_organization, a.added_at, a.is_public
            FROM agents a
            LEFT JOIN entitlements e ON a.id = e.agent_id
            WHERE (a.is_public = TRUE OR e.client_id = ?)
        '''

        params = [client_id]

        # Add text search
        if search_request.query:
            base_query += ''' AND (
                a.name LIKE ? OR 
                a.description LIKE ? OR 
                a.tags LIKE ? OR
                a.provider_organization LIKE ?
            )'''
            search_term = f"%{search_request.query}%"
            params.extend([search_term, search_term, search_term, search_term])

        # Add tag filters
        if search_request.tags:
            for tag in search_request.tags:
                base_query += " AND a.tags LIKE ?"
                params.append(f"%{tag}%")

        # Add provider filter
        if search_request.provider:
            base_query += " AND a.provider_organization LIKE ?"
            params.append(f"%{search_request.provider}%")

        # Add pagination
        base_query += " ORDER BY a.added_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(base_query, params)

        agents = []
        for row in cursor.fetchall():
            agent_id, name, description, url, tags_str, provider, added_at, is_public = row
            tags = tags_str.split(',') if tags_str else []

            agents.append(AgentResource(
                id=agent_id,
                name=name,
                description=description,
                location=AgentLocation(url=url),
                tags=tags,
                is_public=bool(is_public),
                provider_organization=provider,
                added_at=datetime.fromisoformat(added_at) if added_at else None
            ))

        # Get total count for search
        count_query = base_query.replace(
            "SELECT DISTINCT a.id, a.name, a.description, a.agent_card_url, a.tags, a.provider_organization, a.added_at, a.is_public",
            "SELECT COUNT(DISTINCT a.id)"
        ).split("ORDER BY")[0]  # Remove ORDER BY and LIMIT for count

        count_params = params[:-2]  # Remove LIMIT and OFFSET params
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]

        conn.close()

        return RegistryResponse(
            resources=agents,
            total_count=total_count,
            offset=offset,
            limit=limit
        )


# Initialize service
registry_service = AgentRegistryService(db_manager)


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "A2A Agent Registry",
        "version": "0.1.0",
        "description": "Agent discovery and entitlement service for A2A protocol",
        "endpoints": {
            "public_discovery": "/agents/public",
            "entitled_agents": "/agents/entitled",
            "agent_search": "/agents/search"
        }
    }


@app.get("/agents/public", response_model=RegistryResponse)
async def get_public_agents(
        skip: int = Query(0, alias="skip", description="Number of records to skip"),
        top: int = Query(100, alias="top", description="Maximum number of records to return")
):
    """
    Open Discovery endpoint - Get all publicly available agents
    No authentication required
    """
    return registry_service.get_public_agents(offset=skip, limit=top)


@app.get("/agents/entitled", response_model=RegistryResponse)
async def get_entitled_agents(
        skip: int = Query(0, alias="skip", description="Number of records to skip"),
        top: int = Query(100, alias="top", description="Maximum number of records to return"),
        client_info: Dict = Depends(get_current_client)
):
    """
    Get all agents entitled to the authenticated client (includes public agents)
    Requires authentication
    """
    client_id = client_info["client_id"]
    return registry_service.get_entitled_agents(client_id, offset=skip, limit=top)


@app.post("/agents/search", response_model=RegistryResponse)
async def search_agents(
        search_request: SearchRequest,
        skip: int = Query(0, alias="skip", description="Number of records to skip"),
        top: int = Query(100, alias="top", description="Maximum number of records to return"),
        client_info: Dict = Depends(get_current_client)
):
    """
    Search agents based on query and filters
    Returns agents accessible to the authenticated client
    Requires authentication
    """
    client_id = client_info["client_id"]
    return registry_service.search_agents(
        client_id=client_id,
        search_request=search_request,
        offset=skip,
        limit=top
    )


# Admin endpoints (for demonstration - would require admin role in production)

@app.post("/admin/agents")
async def add_agent(
        agent_id: str,
        name: str,
        agent_card_url: str,
        description: str = None,
        tags: List[str] = [],
        is_public: bool = False,
        provider_organization: str = None
):
    """Add a new agent to the registry"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO agents (id, name, description, agent_card_url, tags, is_public, provider_organization)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            agent_id, name, description, agent_card_url,
            ','.join(tags), is_public, provider_organization
        ))
        conn.commit()

        return {"message": f"Agent {agent_id} added successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Agent ID already exists")
    finally:
        conn.close()


@app.post("/admin/clients", response_model=ClientResponse)
async def register_client(client_reg: ClientRegistration):
    """Register a new client application"""
    client_id = f"client_{uuid.uuid4().hex[:8]}"
    client_secret = uuid.uuid4().hex

    conn = db_manager.get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO clients (client_id, client_secret, name, description)
        VALUES (?, ?, ?, ?)
    ''', (client_id, client_secret, client_reg.name, client_reg.description))

    conn.commit()
    conn.close()

    return ClientResponse(
        client_id=client_id,
        client_secret=client_secret,
        name=client_reg.name,
        description=client_reg.description,
        created_at=datetime.now()
    )


@app.post("/admin/entitlements")
async def grant_entitlements(entitlement: EntitlementRequest):
    """Grant agent access to a client"""
    conn = db_manager.get_connection()
    cursor = conn.cursor()

    granted = []
    for agent_id in entitlement.agent_ids:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO entitlements (client_id, agent_id)
                VALUES (?, ?)
            ''', (entitlement.client_id, agent_id))
            granted.append(agent_id)
        except Exception as e:
            logger.error(f"Failed to grant entitlement for agent {agent_id}: {e}")

    conn.commit()
    conn.close()

    return {"message": f"Granted entitlements", "granted_agents": granted}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)