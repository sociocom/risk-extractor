FROM python:3.12-slim-bookworm
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

WORKDIR /app
COPY pyproject.toml uv.lock /app/

# Ensure the installed binary is on the `PATH`
# ENV PATH="/app/.venv/bin:$PATH"
ENV PATH="/root/.local/bin/:$PATH"

# /app/.venv にライブラリがインストールされる
RUN uv sync --frozen --no-cache

COPY *.py ./
# COPY .streamlit ./

ENV STREAMLIT_THEME_BASE="light"
ENV STREAMLIT_THEME_PRIMARY_COLOR="#FF715B"
ENV STREAMLIT_THEME_BACKGROUND_COLOR="#FFFFFF"
ENV STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#34C3B5"
ENV STREAMLIT_THEME_TEXT_COLOR="#4C5454"
ENV STREAMLIT_THEME_FONT="sans serif"
ENV STREAMLIT_THEME_SIDEBAR_BACKGROUND_COLOR="#FF715B"
ENV STREAMLIT_THEME_SIDEBAR_CONTRAST=1.2
ENV STREAMLIT_BROWSER_SERVER_ADDRESS="aoi.naist.jp"
ENV STREAMLIT_BROWSER_SERVER_PORT=7000

ENTRYPOINT ["uv", "run", "streamlit", "run"]

# CMD ["app.py", "--server.port", "7000", "--server.baseUrlPath=/riskun"]