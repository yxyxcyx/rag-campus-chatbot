#!/bin/bash
# dev-stop.sh - Stop all development services

echo "🛑 Stopping RAG Development Environment"
echo "======================================="

docker compose -f docker-compose.dev.yml down

echo " All services stopped"
echo ""
echo " To see container status:"
echo "   docker ps -a"
echo ""
echo "🗑️  To clean up volumes (  deletes data):"
echo "   docker compose -f docker-compose.dev.yml down -v"
