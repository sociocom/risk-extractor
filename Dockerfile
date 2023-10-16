# Multiple stage build
# to generate requirements.txt
FROM python:3.11-slim as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Below is the actual Dockerfile
FROM python:3.11.4

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=requirements-stage /tmp/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# COPY pyproject.toml ./

# RUN pip install poetry
# RUN poetry install

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

ENTRYPOINT ["streamlit", "run"]

# CMD ["app.py", "--server.port", "7000", "--server.baseUrlPath=/riskun"]