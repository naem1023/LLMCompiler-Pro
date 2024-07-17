SHELL = bash
ifneq ("$(wildcard .env)","")
	include .env
endif

.ONESHELL:
up:
	docker compose -f docker-compose.yaml up -d

.ONESHELL:
down:
	docker compose -f docker-compose.yaml down
