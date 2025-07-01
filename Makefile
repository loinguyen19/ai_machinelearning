
.PHONY: build
build:
	docker-compose build --no-cache


.PHONY: up
up:
	docker-compose up -d
	docker-compose logs dispatch_monitor