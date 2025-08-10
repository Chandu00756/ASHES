"""
Database session management and utilities.

Handles SQLAlchemy session creation, connection pooling, and database operations.
"""

from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

from ..core.config import get_config
from .models import Base, Experiment, User, DeviceState, KnowledgeBase

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._initialized:
            return
        
        # Create database engine
        self.engine = create_engine(
            self.config.database.postgres_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=20,
            max_overflow=30,
            echo=self.config.debug,
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self._initialized = True
        logger.info("Database manager initialized")
    
    def create_all_tables(self) -> None:
        """Create all database tables."""
        if not self._initialized:
            self.initialize()
        
        Base.metadata.create_all(bind=self.engine)
        logger.info("All database tables created")
    
    def drop_all_tables(self) -> None:
        """Drop all database tables - USE WITH CAUTION!"""
        if not self._initialized:
            self.initialize()
        
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if not self._initialized:
            self.initialize()
        
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for getting database session."""
    with db_manager.session_scope() as session:
        yield session


def init_db() -> None:
    """Initialize database tables."""
    db_manager.initialize()
    db_manager.create_all_tables()


class DatabaseRepository:
    """Base repository class for database operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save(self, obj) -> None:
        """Save object to database."""
        self.session.add(obj)
        self.session.flush()
    
    def delete(self, obj) -> None:
        """Delete object from database."""
        self.session.delete(obj)
        self.session.flush()
    
    def commit(self) -> None:
        """Commit current transaction."""
        self.session.commit()
    
    def rollback(self) -> None:
        """Rollback current transaction."""
        self.session.rollback()


class ExperimentRepository(DatabaseRepository):
    """Repository for experiment-related database operations."""
    
    def get_by_id(self, experiment_id: str) -> Optional["Experiment"]:
        """Get experiment by ID."""
        from .models import Experiment
        return self.session.query(Experiment).filter(Experiment.id == experiment_id).first()
    
    def get_by_status(self, status: str) -> list["Experiment"]:
        """Get experiments by status."""
        from .models import Experiment
        return self.session.query(Experiment).filter(Experiment.status == status).all()
    
    def get_active_experiments(self) -> list["Experiment"]:
        """Get all active experiments."""
        from .models import Experiment
        return self.session.query(Experiment).filter(
            Experiment.status.in_(['running', 'paused', 'waiting'])
        ).all()
    
    def create_experiment(self, **kwargs) -> "Experiment":
        """Create new experiment."""
        from .models import Experiment
        experiment = Experiment(**kwargs)
        self.save(experiment)
        return experiment


class UserRepository(DatabaseRepository):
    """Repository for user-related database operations."""
    
    def get_by_username(self, username: str) -> Optional["User"]:
        """Get user by username."""
        from .models import User
        return self.session.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional["User"]:
        """Get user by email."""
        from .models import User
        return self.session.query(User).filter(User.email == email).first()
    
    def create_user(self, **kwargs) -> "User":
        """Create new user."""
        from .models import User
        user = User(**kwargs)
        self.save(user)
        return user


class DeviceRepository(DatabaseRepository):
    """Repository for device-related database operations."""
    
    def get_device_state(self, device_id: str) -> Optional["DeviceState"]:
        """Get device state by ID."""
        from .models import DeviceState
        return self.session.query(DeviceState).filter(DeviceState.device_id == device_id).first()
    
    def get_active_devices(self) -> list["DeviceState"]:
        """Get all connected devices."""
        from .models import DeviceState
        return self.session.query(DeviceState).filter(DeviceState.connected == True).all()
    
    def update_device_state(self, device_id: str, **kwargs) -> "DeviceState":
        """Update device state."""
        from .models import DeviceState
        device = self.get_device_state(device_id)
        if device:
            for key, value in kwargs.items():
                setattr(device, key, value)
            self.save(device)
        else:
            device = DeviceState(device_id=device_id, **kwargs)
            self.save(device)
        return device


class KnowledgeRepository(DatabaseRepository):
    """Repository for knowledge base operations."""
    
    def search_by_domain(self, domain: str, limit: int = 10) -> list["KnowledgeBase"]:
        """Search knowledge base by domain."""
        from .models import KnowledgeBase
        return self.session.query(KnowledgeBase).filter(
            KnowledgeBase.domain == domain
        ).order_by(KnowledgeBase.confidence_score.desc()).limit(limit).all()
    
    def add_knowledge(self, **kwargs) -> "KnowledgeBase":
        """Add new knowledge entry."""
        from .models import KnowledgeBase
        knowledge = KnowledgeBase(**kwargs)
        self.save(knowledge)
        return knowledge
