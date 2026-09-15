FROM node:22-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN npm install -g corepack@latest \
    && corepack pnpm install \
    && corepack pnpm run build \
    && pip3 install --break-system-packages --no-cache-dir -r requirements.txt

ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-3000}"]
