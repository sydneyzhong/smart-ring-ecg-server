# --------------------------------
# 基础阶段
# --------------------------------
FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/base:ubuntu22.04 AS base

# 环境变量
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# 清理所有默认源配置
RUN rm -rf /etc/apt/sources.list.d/* && \
    rm -f /etc/apt/sources.list

# 复制本地APT仓库
COPY local-apt-repo /var/local-apt-repo

# 配置本地APT源
RUN echo "deb [trusted=yes] file:/var/local-apt-repo/packages/amd64 ./" > /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check

# 配置时区（单次执行）
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# --------------------------------
# 构建阶段（安装系统依赖）
# --------------------------------
FROM base AS builder

# 安装系统包
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3.10 \
        python3-minimal \
        python3-pip \
        liblapack-dev \
        libopenblas-dev \
        libopenblas0-pthread \
        gfortran \
        fonts-wqy-microhei \
        ca-certificates \
        python3-pkg-resources
# --------------------------------
# 最终镜像阶段
# --------------------------------
FROM base

# 从构建阶段复制已安装的软件
COPY --from=builder /usr /usr
COPY --from=builder /lib /lib
COPY --from=builder /bin /bin
COPY --from=builder /etc /etc

# 设置工作目录
WORKDIR /app

# 复制Python依赖
COPY pypi-packages /pypi-packages
COPY requirements.txt .

# 验证依赖文件
RUN ls -l /pypi-packages && \ 
    ls -l /pypi-packages/Flask-2.0.3-py3-none-any.whl

# 安装Python依赖
RUN /usr/bin/python3 -m pip install --no-index --find-links=/pypi-packages \
    --upgrade pip setuptools wheel && \
    /usr/bin/pip3 install --no-index --find-links=/pypi-packages \
    -r requirements.txt

# 复制应用代码
COPY . .

# 设置目录权限
RUN mkdir -p /app/static /app/reports /tmp/uploads && \
    chmod -R 777 /app/static /app/reports /tmp/uploads

# 暴露端口和启动命令
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]