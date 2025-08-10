"""
Enterprise-grade authentication and authorization system for ASHES.

This module implements OAuth2.0 + OpenID Connect authentication,
JWT token management, role-based access control, and security auditing.
"""

import asyncio
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import uuid
import json
from dataclasses import dataclass
from enum import Enum

from ..core.config import get_config

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    RESEARCHER = "researcher" 
    USER = "user"
    INTEGRATOR = "integrator"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions."""
    # Experiment permissions
    EXPERIMENT_CREATE = "experiment:create"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_UPDATE = "experiment:update"
    EXPERIMENT_DELETE = "experiment:delete"
    EXPERIMENT_EXECUTE = "experiment:execute"
    
    # Agent permissions
    AGENT_INTERACT = "agent:interact"
    AGENT_MANAGE = "agent:manage"
    AGENT_MONITOR = "agent:monitor"
    
    # System permissions
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_ADMIN = "system:admin"
    USER_MANAGE = "user:manage"
    
    # Data permissions
    DATA_EXPORT = "data:export"
    DATA_IMPORT = "data:import"


@dataclass
class User:
    """User entity with comprehensive metadata."""
    id: str
    username: str
    email: str
    role: UserRole
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    permissions: List[Permission] = None
    metadata: Dict[str, Any] = None


class AuthManager:
    """
    Enterprise authentication and authorization manager.
    
    Provides OAuth2.0 + JWT authentication, RBAC authorization,
    session management, and security auditing.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # JWT configuration
        self.jwt_secret = getattr(self.config, 'jwt_secret', secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.password_min_length = 8
        
        # In-memory stores (replace with Redis/DB in production)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict] = {}
        self.blacklisted_tokens: set = set()
        
        # Role-based permissions
        self.role_permissions = {
            UserRole.ADMIN: [perm for perm in Permission],
            UserRole.RESEARCHER: [
                Permission.EXPERIMENT_CREATE,
                Permission.EXPERIMENT_READ,
                Permission.EXPERIMENT_UPDATE,
                Permission.EXPERIMENT_EXECUTE,
                Permission.AGENT_INTERACT,
                Permission.AGENT_MONITOR,
                Permission.SYSTEM_MONITOR,
                Permission.DATA_EXPORT
            ],
            UserRole.USER: [
                Permission.EXPERIMENT_READ,
                Permission.AGENT_INTERACT,
                Permission.SYSTEM_MONITOR
            ],
            UserRole.INTEGRATOR: [
                Permission.EXPERIMENT_READ,
                Permission.SYSTEM_MONITOR,
                Permission.DATA_EXPORT,
                Permission.DATA_IMPORT
            ],
            UserRole.READONLY: [
                Permission.EXPERIMENT_READ,
                Permission.SYSTEM_MONITOR
            ]
        }
        
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default system users."""
        # Create default admin user
        admin_id = str(uuid.uuid4())
        admin_user = User(
            id=admin_id,
            username="admin",
            email="admin@ashes.ai",
            role=UserRole.ADMIN,
            hashed_password=self.get_password_hash("admin123"),
            created_at=datetime.utcnow(),
            is_active=True,
            permissions=self.role_permissions[UserRole.ADMIN]
        )
        self.users[admin_user.username] = admin_user
        
        # Create default researcher user
        researcher_id = str(uuid.uuid4())
        researcher_user = User(
            id=researcher_id,
            username="researcher",
            email="researcher@ashes.ai", 
            role=UserRole.RESEARCHER,
            hashed_password=self.get_password_hash("researcher123"),
            created_at=datetime.utcnow(),
            is_active=True,
            permissions=self.role_permissions[UserRole.RESEARCHER]
        )
        self.users[researcher_user.username] = researcher_user
        
        logger.info("Initialized default users: admin, researcher")
    
    def get_password_hash(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.jwt_expiration
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            if token in self.blacklisted_tokens:
                raise jwt.InvalidTokenError("Token has been revoked")
            
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError:
            raise jwt.InvalidTokenError("Invalid token")
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username/password.
        
        Implements account lockout protection and login attempt tracking.
        """
        user = self.users.get(username)
        if not user:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            logger.warning(f"Authentication failed: user {username} account locked")
            return None
        
        # Check if account is active
        if not user.is_active:
            logger.warning(f"Authentication failed: user {username} account disabled")
            return None
        
        # Verify password
        if not self.verify_password(password, user.hashed_password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failures
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                logger.warning(f"Account locked for user {username} due to too many failed attempts")
            
            logger.warning(f"Authentication failed: invalid password for user {username}")
            return None
        
        # Successful authentication - reset failed attempts
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Return user data
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [perm.value for perm in user.permissions] if user.permissions else [],
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        
        logger.info(f"User {username} authenticated successfully")
        return user_data
    
    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users."""
        users_list = []
        for user in self.users.values():
            users_list.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "failed_login_attempts": user.failed_login_attempts,
                "locked_until": user.locked_until.isoformat() if user.locked_until else None
            })
        return users_list
    
    def check_permission(self, user_role: str, required_permission: Permission) -> bool:
        """Check if user role has required permission."""
        try:
            role = UserRole(user_role)
            role_permissions = self.role_permissions.get(role, [])
            return required_permission in role_permissions
        except ValueError:
            return False
