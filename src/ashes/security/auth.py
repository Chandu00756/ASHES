"""
Authentication and authorization for ASHES system.

Implements JWT-based authentication with role-based access control.
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from ..core.config import get_config
from ..core.logging import get_logger
from ..database.models import User
from ..database.session import get_db


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
logger = get_logger(__name__)


class SecurityManager:
    """Handles authentication and authorization."""
    
    def __init__(self):
        self.config = get_config()
        self.secret_key = self.config.security.secret_key
        self.algorithm = self.config.security.algorithm
        self.access_token_expire_minutes = self.config.security.access_token_expire_minutes
    
    def create_access_token(self, data: Dict[str, any]) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    async def authenticate_user(self, username: str, password: str, db: Session = None) -> Optional[Dict[str, any]]:
        """Authenticate a user with username and password."""
        if not db:
            # For demo purposes, use hardcoded admin user
            if username == "admin" and password == "admin123":
                return {
                    "username": "admin",
                    "email": "admin@ashes-lab.com",
                    "role": "admin",
                    "permissions": ["read", "write", "execute", "admin"]
                }
            elif username == "scientist" and password == "science123":
                return {
                    "username": "scientist",
                    "email": "scientist@ashes-lab.com",
                    "role": "scientist",
                    "permissions": ["read", "write", "execute"]
                }
            return None
        
        # Database-based authentication
        user = db.query(User).filter(User.username == username).first()
        if not user or not self.verify_password(password, user.hashed_password):
            return None
        
        return {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "permissions": user.permissions
        }
    
    def check_permission(self, user: Dict[str, any], required_permission: str) -> bool:
        """Check if user has required permission."""
        user_permissions = user.get("permissions", [])
        return required_permission in user_permissions or "admin" in user_permissions


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, any]:
    """Get current authenticated user from JWT token."""
    security_manager = SecurityManager()
    payload = security_manager.verify_token(token)
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For demo purposes, return user info based on username
    if username == "admin":
        return {
            "username": "admin",
            "email": "admin@ashes-lab.com",
            "role": "admin",
            "permissions": ["read", "write", "execute", "admin"]
        }
    elif username == "scientist":
        return {
            "username": "scientist",
            "email": "scientist@ashes-lab.com",
            "role": "scientist",
            "permissions": ["read", "write", "execute"]
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="User not found",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(current_user: Dict[str, any] = Depends(get_current_user)):
        security_manager = SecurityManager()
        if not security_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker


def require_role(role: str):
    """Decorator to require specific role."""
    def role_checker(current_user: Dict[str, any] = Depends(get_current_user)):
        if current_user.get("role") != role and current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    return role_checker


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_minutes = window_minutes
        self.requests: Dict[str, List[datetime]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.window_minutes)
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


# Rate limiter instances
api_rate_limiter = RateLimiter(max_requests=1000, window_minutes=1)
auth_rate_limiter = RateLimiter(max_requests=10, window_minutes=1)


def check_rate_limit(identifier: str, limiter: RateLimiter = api_rate_limiter):
    """Check rate limit for an identifier."""
    if not limiter.is_allowed(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
