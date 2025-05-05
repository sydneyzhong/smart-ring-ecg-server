# 使用阿里云镜像源
# FROM registry.cn-hangzhou.aliyuncs.com/library/python:3.9-slim
#FROM registry.cn-hangzhou.aliyuncs.com/python:3.9-slim
FROM python:3.9-slim

WORKDIR /app

# 设置时区和安装依赖
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    fonts-wqy-zenhei \ 
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN mkdir -p /app/static && \
    mkdir -p /app/data/ecg_uploads && \
    mkdir -p /app/reports && \
    chmod -R 777 /app/static && \
    chmod -R 777 /app/data && \
    chmod -R 777 /app/reports

# 使用清华pip源加速安装
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

RUN mkdir -p /tmp/uploads

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
