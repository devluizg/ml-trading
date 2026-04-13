FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update -q && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código do projeto
COPY . .

# Diretórios persistentes (serão sobrescritos pelos volumes do Coolify)
RUN mkdir -p data/history journal models logs

# Health check na porta 8080
EXPOSE 8080

ENTRYPOINT ["bash", "entrypoint.sh"]
