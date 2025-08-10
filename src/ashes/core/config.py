"""Configuration management for ASHES system."""

import os
from typing import Optional, List
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="ashes", env="POSTGRES_DB")
    postgres_user: str = Field(default="ashes", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    influxdb_host: str = Field(default="localhost", env="INFLUXDB_HOST")
    influxdb_port: int = Field(default=8086, env="INFLUXDB_PORT")
    influxdb_token: str = Field(default="", env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="ashes-lab", env="INFLUXDB_ORG")
    
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="", env="NEO4J_PASSWORD")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class AIConfig(BaseSettings):
    """AI model configuration settings."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    together_api_key: str = Field(default="", env="TOGETHER_API_KEY")
    
    pinecone_api_key: str = Field(default="", env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-west1-gcp-free", env="PINECONE_ENVIRONMENT")
    
    default_model: str = Field(default="gpt-4-turbo", env="DEFAULT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")
    
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")


class LaboratoryConfig(BaseSettings):
    """Laboratory automation configuration."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    lab_network_cidr: str = Field(default="192.168.1.0/24", env="LAB_NETWORK_CIDR")
    safety_timeout: int = Field(default=30, env="SAFETY_TIMEOUT")
    emergency_stop_enabled: bool = Field(default=True, env="EMERGENCY_STOP_ENABLED")
    
    robot_controllers: List[str] = Field(
        default=["192.168.1.100", "192.168.1.101"], 
        env="ROBOT_CONTROLLERS"
    )
    
    analytical_instruments: List[str] = Field(
        default=["192.168.1.102", "192.168.1.103"], 
        env="ANALYTICAL_INSTRUMENTS"
    )


class SecurityConfig(BaseSettings):
    """Security and authentication configuration."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    secret_key: str = Field(default="", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    encryption_key: str = Field(default="", env="ENCRYPTION_KEY")
    
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")
    max_login_attempts: int = Field(default=3, env="MAX_LOGIN_ATTEMPTS")


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    prometheus_host: str = Field(default="localhost", env="PROMETHEUS_HOST")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    grafana_host: str = Field(default="localhost", env="GRAFANA_HOST")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    
    jaeger_host: str = Field(default="localhost", env="JAEGER_HOST")
    jaeger_port: int = Field(default=14268, env="JAEGER_PORT")
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    structured_logging: bool = Field(default=True, env="STRUCTURED_LOGGING")


class ASHESConfig(BaseSettings):
    """Main ASHES system configuration."""
    
    model_config = ConfigDict(env_file=".env", extra="ignore")
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    worker_concurrency: int = Field(default=4, env="WORKER_CONCURRENCY")
    max_experiments_concurrent: int = Field(default=10, env="MAX_EXPERIMENTS_CONCURRENT")
    
    data_retention_days: int = Field(default=365, env="DATA_RETENTION_DAYS")
    backup_enabled: bool = Field(default=True, env="BACKUP_ENABLED")
    
    database: DatabaseConfig = DatabaseConfig()
    ai: AIConfig = AIConfig()
    laboratory: LaboratoryConfig = LaboratoryConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()


# Global configuration instance
config = ASHESConfig()


def get_config() -> ASHESConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> ASHESConfig:
    """Reload configuration from environment and .env file."""
    global config
    config = ASHESConfig()
    return config
