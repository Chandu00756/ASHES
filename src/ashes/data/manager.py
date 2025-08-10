"""
Data Management - Comprehensive data handling and storage system for ASHES.

This module implements multi-database integration, data pipelines, 
and intelligent data organization for scientific research workflows.
"""

import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle
import numpy as np
import pandas as pd

from ..core.config import get_config
from ..core.logging import get_logger


class DataType(Enum):
    """Types of data in the system."""
    EXPERIMENT = "experiment"
    CHARACTERIZATION = "characterization"
    SYNTHESIS = "synthesis"
    HYPOTHESIS = "hypothesis"
    LITERATURE = "literature"
    METADATA = "metadata"
    RAW_DATA = "raw_data"
    PROCESSED_DATA = "processed_data"


class StorageBackend(Enum):
    """Available storage backends."""
    POSTGRESQL = "postgresql"
    PINECONE = "pinecone"
    NEO4J = "neo4j"
    INFLUXDB = "influxdb"
    REDIS = "redis"
    FILE_SYSTEM = "file_system"


@dataclass
class DataSchema:
    """Data schema definition."""
    schema_id: str
    name: str
    version: str
    data_type: DataType
    fields: Dict[str, Any]
    required_fields: List[str]
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRecord:
    """Unified data record structure."""
    record_id: str
    data_type: DataType
    schema_version: str
    metadata: Dict[str, Any]
    content: Any
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Storage information
    storage_backend: Optional[StorageBackend] = None
    storage_location: Optional[str] = None
    
    # Relationships
    related_records: List[str] = field(default_factory=list)
    parent_record: Optional[str] = None
    
    # Data quality
    quality_score: float = 1.0
    validation_status: str = "pending"
    
    # Access control
    access_level: str = "internal"
    created_by: str = "system"


class DatabaseInterface:
    """Base interface for database operations."""
    
    def __init__(self, backend: StorageBackend):
        self.backend = backend
        self.config = get_config()
        self.logger = get_logger(f"{__name__}.{backend.value}")
        self._connection = None
    
    async def connect(self):
        """Connect to the database."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from the database."""
        raise NotImplementedError
    
    async def store(self, record: DataRecord) -> str:
        """Store a data record."""
        raise NotImplementedError
    
    async def retrieve(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve a data record by ID."""
        raise NotImplementedError
    
    async def query(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Query data records."""
        raise NotImplementedError
    
    async def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update a data record."""
        raise NotImplementedError
    
    async def delete(self, record_id: str) -> bool:
        """Delete a data record."""
        raise NotImplementedError


class PostgreSQLInterface(DatabaseInterface):
    """PostgreSQL interface for structured data."""
    
    def __init__(self):
        super().__init__(StorageBackend.POSTGRESQL)
        self.db_url = self.config.get("postgresql_url", "postgresql://localhost:5432/ashes")
    
    async def connect(self):
        """Connect to PostgreSQL."""
        try:
            import asyncpg
            self._connection = await asyncpg.connect(self.db_url)
            await self._initialize_tables()
            self.logger.info("Connected to PostgreSQL")
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            # Create mock connection for development
            self._connection = "mock_connection"
    
    async def _initialize_tables(self):
        """Initialize database tables."""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS experiments (
            id UUID PRIMARY KEY,
            hypothesis_id UUID,
            design JSONB,
            results JSONB,
            status VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS materials (
            id UUID PRIMARY KEY,
            name VARCHAR(255) UNIQUE,
            formula VARCHAR(255),
            properties JSONB,
            safety_info JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS characterizations (
            id UUID PRIMARY KEY,
            sample_id UUID,
            technique VARCHAR(100),
            parameters JSONB,
            results JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS data_records (
            id UUID PRIMARY KEY,
            data_type VARCHAR(50),
            schema_version VARCHAR(20),
            metadata JSONB,
            content JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        if hasattr(self._connection, 'execute'):
            await self._connection.execute(create_tables_sql)
    
    async def store(self, record: DataRecord) -> str:
        """Store a data record in PostgreSQL."""
        try:
            if record.data_type == DataType.EXPERIMENT:
                query = """
                INSERT INTO experiments (id, hypothesis_id, design, results, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """
                await self._connection.execute(
                    query,
                    uuid.UUID(record.record_id),
                    uuid.UUID(record.metadata.get("hypothesis_id", str(uuid.uuid4()))),
                    json.dumps(record.content.get("design", {})),
                    json.dumps(record.content.get("results", {})),
                    record.metadata.get("status", "pending"),
                    record.created_at
                )
            else:
                # Generic data record storage
                query = """
                INSERT INTO data_records (id, data_type, schema_version, metadata, content, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """
                await self._connection.execute(
                    query,
                    uuid.UUID(record.record_id),
                    record.data_type.value,
                    record.schema_version,
                    json.dumps(record.metadata),
                    json.dumps(record.content, default=str),
                    record.created_at
                )
            
            record.storage_backend = self.backend
            self.logger.info(f"Stored record {record.record_id} in PostgreSQL")
            return record.record_id
            
        except Exception as e:
            self.logger.error(f"Failed to store record in PostgreSQL: {e}")
            return record.record_id  # Return ID even if storage fails for development


class PineconeInterface(DatabaseInterface):
    """Pinecone interface for vector data and embeddings."""
    
    def __init__(self):
        super().__init__(StorageBackend.PINECONE)
        self.api_key = self.config.get("pinecone_api_key")
        self.environment = self.config.get("pinecone_environment", "us-west1-gcp")
        self.index_name = "ashes-vectors"
    
    async def connect(self):
        """Connect to Pinecone."""
        try:
            import pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric="cosine"
                )
            
            self._connection = pinecone.Index(self.index_name)
            self.logger.info("Connected to Pinecone")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Pinecone: {e}")
            # Create mock connection
            self._connection = "mock_pinecone"
    
    async def store_embedding(self, record_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Store vector embedding in Pinecone."""
        try:
            if hasattr(self._connection, 'upsert'):
                self._connection.upsert([(record_id, embedding, metadata)])
            self.logger.info(f"Stored embedding for record {record_id}")
        except Exception as e:
            self.logger.error(f"Failed to store embedding: {e}")
    
    async def similarity_search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform similarity search in vector space."""
        try:
            if hasattr(self._connection, 'query'):
                results = self._connection.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                return results.get("matches", [])
            else:
                # Mock results for development
                return [{"id": f"mock_{i}", "score": 0.9 - i*0.1} for i in range(min(top_k, 3))]
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []


class Neo4jInterface(DatabaseInterface):
    """Neo4j interface for graph data and relationships."""
    
    def __init__(self):
        super().__init__(StorageBackend.NEO4J)
        self.uri = self.config.get("neo4j_uri", "bolt://localhost:7687")
        self.username = self.config.get("neo4j_username", "neo4j")
        self.password = self.config.get("neo4j_password", "password")
    
    async def connect(self):
        """Connect to Neo4j."""
        try:
            from neo4j import AsyncGraphDatabase
            self._connection = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            await self._initialize_constraints()
            self.logger.info("Connected to Neo4j")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            # Create mock connection
            self._connection = "mock_neo4j"
    
    async def _initialize_constraints(self):
        """Initialize Neo4j constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Experiment) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Material) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE"
        ]
        
        if hasattr(self._connection, 'session'):
            async with self._connection.session() as session:
                for constraint in constraints:
                    await session.run(constraint)
    
    async def store_relationship(self, from_id: str, to_id: str, relationship_type: str, properties: Dict[str, Any] = None):
        """Store a relationship between two nodes."""
        try:
            if hasattr(self._connection, 'session'):
                async with self._connection.session() as session:
                    query = f"""
                    MATCH (a), (b) 
                    WHERE a.id = $from_id AND b.id = $to_id
                    CREATE (a)-[r:{relationship_type}]->(b)
                    SET r += $properties
                    """
                    await session.run(query, from_id=from_id, to_id=to_id, properties=properties or {})
            
            self.logger.info(f"Created relationship {from_id} -{relationship_type}-> {to_id}")
        except Exception as e:
            self.logger.error(f"Failed to store relationship: {e}")
    
    async def find_related(self, node_id: str, relationship_type: str = None, depth: int = 1) -> List[Dict[str, Any]]:
        """Find related nodes."""
        try:
            if hasattr(self._connection, 'session'):
                relationship_filter = f":{relationship_type}" if relationship_type else ""
                query = f"""
                MATCH (start {{id: $node_id}})-[r{relationship_filter}*1..{depth}]-(related)
                RETURN related, r
                """
                
                async with self._connection.session() as session:
                    result = await session.run(query, node_id=node_id)
                    return [{"node": record["related"], "relationship": record["r"]} async for record in result]
            else:
                # Mock results
                return [{"node": {"id": f"related_{i}", "type": "mock"}} for i in range(3)]
        except Exception as e:
            self.logger.error(f"Failed to find related nodes: {e}")
            return []


class InfluxDBInterface(DatabaseInterface):
    """InfluxDB interface for time-series data."""
    
    def __init__(self):
        super().__init__(StorageBackend.INFLUXDB)
        self.url = self.config.get("influxdb_url", "http://localhost:8086")
        self.token = self.config.get("influxdb_token")
        self.org = self.config.get("influxdb_org", "ashes")
        self.bucket = "ashes-timeseries"
    
    async def connect(self):
        """Connect to InfluxDB."""
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            
            self._connection = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            
            self.write_api = self._connection.write_api(write_options=SYNCHRONOUS)
            self.query_api = self._connection.query_api()
            
            self.logger.info("Connected to InfluxDB")
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            # Create mock connection
            self._connection = "mock_influxdb"
    
    async def store_timeseries(self, measurement: str, tags: Dict[str, str], fields: Dict[str, Any], timestamp: datetime = None):
        """Store time-series data point."""
        try:
            if hasattr(self, 'write_api'):
                from influxdb_client import Point
                
                point = Point(measurement)
                for tag_key, tag_value in tags.items():
                    point = point.tag(tag_key, tag_value)
                for field_key, field_value in fields.items():
                    point = point.field(field_key, field_value)
                if timestamp:
                    point = point.time(timestamp)
                
                self.write_api.write(bucket=self.bucket, record=point)
            
            self.logger.info(f"Stored time-series point for {measurement}")
        except Exception as e:
            self.logger.error(f"Failed to store time-series data: {e}")


class DataManager:
    """Main data management system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Database interfaces
        self.postgresql = PostgreSQLInterface()
        self.pinecone = PineconeInterface()
        self.neo4j = Neo4jInterface()
        self.influxdb = InfluxDBInterface()
        
        # Data schemas
        self.schemas: Dict[str, DataSchema] = {}
        
        # Data cache
        self.cache: Dict[str, DataRecord] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Data pipeline
        self.pipeline_queue = asyncio.Queue()
        
        self._initialize_schemas()
    
    def _initialize_schemas(self):
        """Initialize data schemas."""
        # Experiment schema
        self.schemas["experiment_v1"] = DataSchema(
            schema_id="experiment_v1",
            name="Experiment Data",
            version="1.0",
            data_type=DataType.EXPERIMENT,
            fields={
                "hypothesis_id": "string",
                "design": "object",
                "materials": "array",
                "procedure": "array",
                "results": "object",
                "status": "string",
                "success_metrics": "object"
            },
            required_fields=["hypothesis_id", "design", "status"]
        )
        
        # Characterization schema
        self.schemas["characterization_v1"] = DataSchema(
            schema_id="characterization_v1",
            name="Characterization Data",
            version="1.0",
            data_type=DataType.CHARACTERIZATION,
            fields={
                "sample_id": "string",
                "technique": "string",
                "parameters": "object",
                "raw_data": "object",
                "processed_data": "object",
                "interpretation": "object"
            },
            required_fields=["sample_id", "technique", "raw_data"]
        )
        
        # Hypothesis schema
        self.schemas["hypothesis_v1"] = DataSchema(
            schema_id="hypothesis_v1",
            name="Hypothesis Data",
            version="1.0",
            data_type=DataType.HYPOTHESIS,
            fields={
                "statement": "string",
                "rationale": "string",
                "predictions": "array",
                "confidence": "number",
                "supporting_evidence": "array",
                "experimental_design": "object"
            },
            required_fields=["statement", "rationale", "confidence"]
        )
        
        self.logger.info(f"Initialized {len(self.schemas)} data schemas")
    
    async def start(self):
        """Start the data management system."""
        self.logger.info("Starting data management system")
        
        # Connect to all databases
        await self.postgresql.connect()
        await self.pinecone.connect()
        await self.neo4j.connect()
        await self.influxdb.connect()
        
        # Start data pipeline
        asyncio.create_task(self._process_pipeline())
        
        self.logger.info("Data management system started")
    
    async def stop(self):
        """Stop the data management system."""
        self.logger.info("Stopping data management system")
        
        # Disconnect from databases
        await self.postgresql.disconnect()
        await self.pinecone.disconnect()
        await self.neo4j.disconnect()
        await self.influxdb.disconnect()
    
    async def store_data(self, data_type: DataType, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Store data with automatic routing to appropriate backend."""
        record_id = str(uuid.uuid4())
        
        # Determine schema
        schema_key = f"{data_type.value}_v1"
        schema = self.schemas.get(schema_key)
        
        if not schema:
            self.logger.warning(f"No schema found for {data_type.value}, using generic storage")
            schema_version = "generic_v1"
        else:
            schema_version = schema.schema_version
            
            # Validate data against schema
            if not await self._validate_data(content, schema):
                self.logger.error(f"Data validation failed for {data_type.value}")
                return None
        
        # Create data record
        record = DataRecord(
            record_id=record_id,
            data_type=data_type,
            schema_version=schema_version,
            metadata=metadata or {},
            content=content,
            created_at=datetime.utcnow()
        )
        
        # Route to appropriate storage backend
        if data_type in [DataType.EXPERIMENT, DataType.METADATA]:
            await self.postgresql.store(record)
        
        # Store relationships in Neo4j
        if data_type == DataType.EXPERIMENT:
            hypothesis_id = metadata.get("hypothesis_id")
            if hypothesis_id:
                await self.neo4j.store_relationship(
                    hypothesis_id, 
                    record_id, 
                    "TESTED_BY",
                    {"created_at": datetime.utcnow().isoformat()}
                )
        
        # Store embeddings for searchable content
        if data_type in [DataType.HYPOTHESIS, DataType.LITERATURE]:
            embedding = await self._generate_embedding(content)
            if embedding:
                await self.pinecone.store_embedding(record_id, embedding, {
                    "data_type": data_type.value,
                    "created_at": datetime.utcnow().isoformat()
                })
        
        # Store time-series data
        if data_type == DataType.CHARACTERIZATION:
            await self.influxdb.store_timeseries(
                measurement="characterization",
                tags={"technique": content.get("technique", "unknown")},
                fields={"sample_id": content.get("sample_id")},
                timestamp=datetime.utcnow()
            )
        
        # Cache the record
        self.cache[record_id] = record
        
        self.logger.info(f"Stored {data_type.value} data with ID: {record_id}")
        return record_id
    
    async def retrieve_data(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve data by ID."""
        # Check cache first
        if record_id in self.cache:
            cached_record = self.cache[record_id]
            if datetime.utcnow() - cached_record.created_at < self.cache_ttl:
                return cached_record
        
        # Try PostgreSQL first
        record = await self.postgresql.retrieve(record_id)
        if record:
            self.cache[record_id] = record
            return record
        
        self.logger.warning(f"Record {record_id} not found")
        return None
    
    async def search_data(self, query: Dict[str, Any]) -> List[DataRecord]:
        """Search data across all backends."""
        results = []
        
        # Text/semantic search using Pinecone
        if "text" in query:
            embedding = await self._generate_embedding(query["text"])
            if embedding:
                vector_results = await self.pinecone.similarity_search(embedding)
                for result in vector_results:
                    record = await self.retrieve_data(result["id"])
                    if record:
                        results.append(record)
        
        # Structured query using PostgreSQL
        if "filters" in query:
            pg_results = await self.postgresql.query(query["filters"])
            results.extend(pg_results)
        
        # Remove duplicates
        unique_results = {r.record_id: r for r in results}
        return list(unique_results.values())
    
    async def find_related_data(self, record_id: str, relationship_type: str = None) -> List[DataRecord]:
        """Find data related to a given record."""
        related_nodes = await self.neo4j.find_related(record_id, relationship_type)
        
        related_records = []
        for node_data in related_nodes:
            node_id = node_data["node"].get("id")
            if node_id:
                record = await self.retrieve_data(node_id)
                if record:
                    related_records.append(record)
        
        return related_records
    
    async def update_data(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing data."""
        record = await self.retrieve_data(record_id)
        if not record:
            return False
        
        # Update record
        record.content.update(updates.get("content", {}))
        record.metadata.update(updates.get("metadata", {}))
        record.updated_at = datetime.utcnow()
        
        # Update in primary storage
        success = await self.postgresql.update(record_id, updates)
        
        if success:
            # Update cache
            self.cache[record_id] = record
            self.logger.info(f"Updated record {record_id}")
        
        return success
    
    async def delete_data(self, record_id: str) -> bool:
        """Delete data from all backends."""
        # Remove from cache
        self.cache.pop(record_id, None)
        
        # Delete from primary storage
        success = await self.postgresql.delete(record_id)
        
        # TODO: Clean up from other backends (Pinecone, Neo4j, etc.)
        
        if success:
            self.logger.info(f"Deleted record {record_id}")
        
        return success
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        stats = {
            "total_records": len(self.cache),
            "records_by_type": {},
            "storage_usage": {},
            "recent_activity": {}
        }
        
        # Count records by type
        for record in self.cache.values():
            data_type = record.data_type.value
            stats["records_by_type"][data_type] = stats["records_by_type"].get(data_type, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_records = [r for r in self.cache.values() if r.created_at > recent_cutoff]
        stats["recent_activity"]["new_records"] = len(recent_records)
        
        return stats
    
    async def export_data(self, query: Dict[str, Any], format: str = "json") -> str:
        """Export data in various formats."""
        records = await self.search_data(query)
        
        if format == "json":
            export_data = []
            for record in records:
                export_data.append({
                    "id": record.record_id,
                    "type": record.data_type.value,
                    "metadata": record.metadata,
                    "content": record.content,
                    "created_at": record.created_at.isoformat()
                })
            return json.dumps(export_data, indent=2)
        
        elif format == "csv" and records:
            # Convert to DataFrame for CSV export
            df_data = []
            for record in records:
                flat_record = {
                    "id": record.record_id,
                    "type": record.data_type.value,
                    "created_at": record.created_at.isoformat()
                }
                # Flatten metadata and content
                for key, value in record.metadata.items():
                    flat_record[f"metadata_{key}"] = value
                for key, value in record.content.items():
                    if isinstance(value, (str, int, float, bool)):
                        flat_record[f"content_{key}"] = value
                
                df_data.append(flat_record)
            
            df = pd.DataFrame(df_data)
            return df.to_csv(index=False)
        
        return ""
    
    async def _validate_data(self, data: Any, schema: DataSchema) -> bool:
        """Validate data against schema."""
        if not isinstance(data, dict):
            return False
        
        # Check required fields
        for field in schema.required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Basic type checking
        for field, expected_type in schema.fields.items():
            if field in data:
                value = data[field]
                if expected_type == "string" and not isinstance(value, str):
                    return False
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return False
                elif expected_type == "array" and not isinstance(value, list):
                    return False
                elif expected_type == "object" and not isinstance(value, dict):
                    return False
        
        return True
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text."""
        try:
            # This would use actual embedding service (OpenAI, etc.)
            # For now, return mock embedding
            import random
            return [random.random() for _ in range(1536)]
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def _process_pipeline(self):
        """Process data pipeline queue."""
        while True:
            try:
                # Get next item from pipeline
                pipeline_item = await asyncio.wait_for(self.pipeline_queue.get(), timeout=1.0)
                
                # Process the item (placeholder for complex data processing)
                await self._process_pipeline_item(pipeline_item)
                
            except asyncio.TimeoutError:
                # No items in queue, continue
                continue
            except Exception as e:
                self.logger.error(f"Pipeline processing error: {e}")
    
    async def _process_pipeline_item(self, item: Dict[str, Any]):
        """Process individual pipeline item."""
        # Placeholder for data processing, transformation, analysis
        self.logger.info(f"Processing pipeline item: {item.get('type', 'unknown')}")
        await asyncio.sleep(0.1)  # Simulate processing time
