#!/bin/bash

# ASHES System Backup Script
# Creates comprehensive backups of all system data

set -e

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🗄️  Creating ASHES system backup in: $BACKUP_DIR"

# Backup databases
echo "📊 Backing up PostgreSQL..."
docker-compose exec -T postgres pg_dump -U ashes ashes > "$BACKUP_DIR/postgres.sql"

echo "📈 Backing up InfluxDB..."
docker-compose exec -T influxdb influx backup /tmp/influx_backup
docker cp $(docker-compose ps -q influxdb):/tmp/influx_backup "$BACKUP_DIR/influxdb"

echo "🕸️  Backing up Neo4j..."
docker-compose exec -T neo4j neo4j-admin backup --backup-dir=/tmp/neo4j_backup --name=ashes
docker cp $(docker-compose ps -q neo4j):/tmp/neo4j_backup "$BACKUP_DIR/neo4j"

# Backup Redis
echo "🔴 Backing up Redis..."
docker-compose exec -T redis redis-cli BGSAVE
sleep 5
docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis.rdb"

# Backup configuration
echo "⚙️  Backing up configuration..."
cp .env "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"
cp -r monitoring "$BACKUP_DIR/"

# Backup logs
echo "📝 Backing up logs..."
mkdir -p "$BACKUP_DIR/logs"
docker-compose logs > "$BACKUP_DIR/logs/all_services.log"

# Create archive
echo "📦 Creating backup archive..."
cd backups
tar -czf "ashes_backup_$(date +%Y%m%d_%H%M%S).tar.gz" "$(basename "$BACKUP_DIR")"
rm -rf "$(basename "$BACKUP_DIR")"

echo "✅ Backup complete: backups/ashes_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
