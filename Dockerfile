FROM python:3.12-slim
WORKDIR /app

# install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --no-root

# copy application code
COPY . .

# run application
CMD ["python", "-u", "main.py"]