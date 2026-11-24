# stt-lab MCP server: read-only access to the meeting index (Postgres + pgvector) for MCP clients.
FROM python:3.12.8-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY mcp_server/requirements.txt mcp_server/requirements.txt
RUN pip install --no-cache-dir -r mcp_server/requirements.txt
COPY mcp_server/ mcp_server/
RUN useradd --system --uid 10001 mcp
USER mcp
EXPOSE 8765
CMD ["python", "-m", "mcp_server.server"]
