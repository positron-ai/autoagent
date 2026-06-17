FROM autoagent-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends bash coreutils findutils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
