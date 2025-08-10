#!/bin/bash

# ASHES System Restore Script
# Restores system from backup

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    echo "Available backups:"
    ls -la backups/*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="./restore_$(date +%Y%m%d_%H%M%S)"

echo "🔄 Restoring ASHES system from: $BACKUP_FILE"

# Extract backup
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR" --strip-components=1

# Stop services
echo "🛑 Stopping services..."
docker-compose down

# Restore configuration
echo "⚙️  Restoring configuration..."
cp "$RESTORE_DIR/.env" .
cp "$RESTORE_DIR/docker-compose.yml" .
cp -r "$RESTORE_DIR/monitoring" .

# Start databases
echo "🗄️  Starting databases..."
docker-compose up -d postgres redis neo4j influxdb
sleep 30

# Restore PostgreSQL
echo "📊 Restoring PostgreSQL..."
docker-compose exec -T postgres dropdb -U ashes ashes --if-exists
docker-compose exec -T postgres createdb -U ashes ashes
docker-compose exec -T postgres psql -U ashes ashes < "$RESTORE_DIR/postgres.sql"

# Restore Redis
echo "🔴 Restoring Redis..."
docker cp "$RESTORE_DIR/redis.rdb" $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis

# Restore Neo4j
echo "🕸️  Restoring Neo4j..."
docker cp "$RESTORE_DIR/neo4j" $(docker-compose ps -q neo4j):/tmp/neo4j_backup
docker-compose exec -T neo4j neo4j-admin restore --from=/tmp/neo4j_backup/ashes --database=neo4j --force

# Restore InfluxDB
echo "📈 Restoring InfluxDB..."
docker cp "$RESTORE_DIR/influxdb" $(docker-compose ps -q influxdb):/tmp/influx_backup
docker-compose exec -T influxdb influx restore /tmp/influx_backup

# Start all services
echo "🚀 Starting all services..."
docker-compose up -d

# Cleanup
rm -rf "$RESTORE_DIR"

echo "✅ Restore complete!"
