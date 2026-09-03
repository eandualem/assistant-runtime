.PHONY: help install dev lint format fix test test-file check clean db-up db-down db-migrate db-upgrade

.DEFAULT_GOAL := help

# --- Directories ---
SRC_DIR := src/assistant_runtime
TESTS_DIR := tests

# --- Color Codes ---
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
CYAN   := $(shell tput -Txterm setaf 6)
RED    := $(shell tput -Txterm setaf 1)
BOLD   := $(shell tput -Txterm bold)
RESET  := $(shell tput -Txterm sgr0)

# --- Setup ---
install:
	@echo "${CYAN}Installing dependencies with uv...${RESET}"
	uv sync --all-extras
	@echo "${GREEN}Install complete.${RESET}"

# --- Development ---
dev:
	@echo "${CYAN}Starting dev server...${RESET}"
	uv run uvicorn assistant_runtime.main:app --reload --host 0.0.0.0 --port $${PORT:-7100}

# --- Code Quality ---
lint:
	@echo "${CYAN}Running linter...${RESET}"
	uv run ruff check $(SRC_DIR) $(TESTS_DIR)
	@echo "${GREEN}Lint complete.${RESET}"

format:
	@echo "${CYAN}Formatting code...${RESET}"
	uv run ruff format $(SRC_DIR) $(TESTS_DIR)
	@echo "${GREEN}Format complete.${RESET}"

fix:
	@echo "${CYAN}Auto-fixing issues...${RESET}"
	uv run ruff check --fix $(SRC_DIR) $(TESTS_DIR)
	uv run ruff format $(SRC_DIR) $(TESTS_DIR)
	@echo "${GREEN}Fix complete.${RESET}"

# --- Testing ---
test:
	@echo "${CYAN}Running tests...${RESET}"
	uv run pytest $(TESTS_DIR) -v
	@echo "${GREEN}Tests complete.${RESET}"

test-file:
	@echo "${CYAN}Running test file: $(FILE)${RESET}"
	uv run pytest $(FILE) -v

# --- CI Gate ---
check:
	@echo "${CYAN}Running full CI gate...${RESET}"
	uv run ruff check $(SRC_DIR) $(TESTS_DIR)
	uv run ruff format --check $(SRC_DIR) $(TESTS_DIR)
	uv run pytest $(TESTS_DIR) -v
	@echo "${GREEN}All checks passed.${RESET}"

# --- Cleanup ---
clean:
	@echo "${CYAN}Cleaning build artifacts...${RESET}"
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "${GREEN}Clean complete.${RESET}"

# --- Database ---
db-up:
	@echo "${CYAN}Starting Postgres...${RESET}"
	docker compose up -d postgres
	@echo "${GREEN}Postgres started on port 5434.${RESET}"

db-down:
	@echo "${CYAN}Stopping Postgres...${RESET}"
	docker compose down
	@echo "${GREEN}Postgres stopped.${RESET}"

db-migrate:
	@echo "${CYAN}Creating migration: $(MSG)${RESET}"
	PYTHONPATH=src uv run alembic revision --autogenerate -m "$(MSG)"
	@echo "${GREEN}Migration created.${RESET}"

db-upgrade:
	@echo "${CYAN}Running migrations...${RESET}"
	PYTHONPATH=src uv run alembic upgrade head
	@echo "${GREEN}Migrations applied.${RESET}"

# --- Help ---
help:
	@echo "${BOLD}${CYAN}Assistant Runtime - Development Commands${RESET}"
	@echo ""
	@echo "  ${GREEN}make install${RESET}            Install dependencies (uv sync)"
	@echo "  ${GREEN}make dev${RESET}                Start dev server (uvicorn --reload)"
	@echo "  ${GREEN}make lint${RESET}               Run linter (ruff check)"
	@echo "  ${GREEN}make format${RESET}             Format code (ruff format)"
	@echo "  ${GREEN}make fix${RESET}                Auto-fix lint + format"
	@echo "  ${GREEN}make test${RESET}               Run all tests"
	@echo "  ${GREEN}make test-file FILE=...${RESET} Run single test file"
	@echo "  ${GREEN}make check${RESET}              Full CI gate (lint + format check + test)"
	@echo "  ${GREEN}make clean${RESET}              Remove build artifacts and caches"
	@echo ""
	@echo "  ${YELLOW}Database:${RESET}"
	@echo "  ${GREEN}make db-up${RESET}              Start Postgres (docker compose)"
	@echo "  ${GREEN}make db-down${RESET}            Stop Postgres"
	@echo "  ${GREEN}make db-migrate MSG=...${RESET} Create new Alembic migration"
	@echo "  ${GREEN}make db-upgrade${RESET}         Run pending migrations"
