FROM python:3.12-slim

# 1) 基础系统依赖（尽量精简）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential \
        git \
        bash \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2) 安装 uv（更快的 Python 包管理与 workspace 支持）
ENV UV_LINK_MODE=copy
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# 3) 复制 .gitmodules 以便初始化 submodules
COPY .gitmodules ./

# 4) 复制必要的 git 配置以支持 submodule 初始化
COPY .git .git

# 5) 初始化并拉取 git submodules（NexA4A 及其内部 submodules）
RUN git submodule update --init --recursive || \
    (cd NexA4A && git submodule update --init --recursive || true)

# 6) 复制依赖清单以进行分层缓存
#   根项目 + 子工作空间（NexA4A 和 Gyrfalcon）
COPY pyproject.toml ./
COPY uv.lock ./
COPY Gyrfalcon/pyproject.toml Gyrfalcon/pyproject.toml
COPY NexA4A/pyproject.toml NexA4A/pyproject.toml
COPY NexA4A/uv.lock NexA4A/uv.lock
COPY NexA4A/nexau/pyproject.toml NexA4A/nexau/pyproject.toml

# 7) 复制全部源码（用户要求直接拷贝，不 clone）
COPY . .

# 8) 同步并安装全部依赖（含 workspace，可重复）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# 9) 默认使用 uv 的虚拟环境
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="/app/.venv/bin:${PATH}"

# 10) 健康检查：打印已安装的 CLI 列表
RUN python -V && pip -V && uv pip list | head -n 50 || true

# 11) 指定默认启动命令（可被 docker run 覆盖）
CMD ["/bin/bash"]
