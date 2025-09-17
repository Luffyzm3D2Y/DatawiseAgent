FROM quay.io/jupyter/docker-stacks-foundation

SHELL ["/bin/bash", "-o", "pipefail", "-c"]


USER root
ENV HOME=/home/root
ENV NB_USER=root
ENV NB_UID=0
ENV NB_GID=0
# 配置 Conda 使用国内镜像源
RUN conda config --system --remove-key channels && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --system --set show_channel_urls yes




# 打印环境变量以检查是否存在代理配置
RUN env


# 安装 lsof 等工具
RUN apt-get update && apt-get install -y \
    lsof \
    curl \
    vim \
    htop \
    tree \
    net-tools \
    unzip \
    file \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mamba install --yes jupyter_kernel_gateway ipykernel && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# 安装机器学习和数据分析常用库
RUN mamba install --yes \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    xgboost \
    openpyxl \
    && mamba clean --all -f -y

# support vision_tool
RUN mamba install --yes \
    openai==1.58.1 \
    pyxlsb \
    && mamba clean --all -f -y

RUN pip install plotly \
    kaleido \
    holoviews \
    ternary \
    regex 



# 安装支持中文的字体
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 配置 Matplotlib 使用中文字体
RUN python -c "import matplotlib; \
from pathlib import Path; \
config_dir = Path(matplotlib.get_configdir()); \
rc_file = config_dir / 'matplotlibrc'; \
config_dir.mkdir(parents=True, exist_ok=True); \
rc_file.write_text('font.family : Noto Sans CJK JP\\n'); \
print('Matplotlib configuration updated at:', rc_file)"

# Create /mnt and adjust permissions
RUN mkdir -p /mnt && \
    chown -R 0:0 /mnt && \
    chmod -R 775 /mnt && \
    fix-permissions /mnt


COPY start.sh /usr/local/bin/start.sh
# 替换 Jupyter Kernel 配置文件，禁用颜色控制
COPY ipython_kernel_config.py /home/root/.ipython/profile_default/

RUN chmod +x /usr/local/bin/start.sh

ENV TOKEN="UNSET"


# CMD ["tail", "-f", "/dev/null"]



EXPOSE 8888

WORKDIR "${HOME}"

ENTRYPOINT ["/usr/local/bin/start.sh"]