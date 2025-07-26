# Makefile for vtoober frontend build and deploy

FRONTEND_DIR=vtoober-frontend
DEPLOY_DIR=frontend

.PHONY: all clean build deploy

all: build deploy

clean:
	rm -rf $(FRONTEND_DIR)/out
	rm -f $(DEPLOY_DIR)/assets/index-*.js $(DEPLOY_DIR)/assets/main-*.js

build:
	cd $(FRONTEND_DIR) && npm install && npm run build

deploy:
	cp -r $(FRONTEND_DIR)/out/renderer/. $(DEPLOY_DIR)/

# Usage:
#   make           # Clean, build, and deploy
#   make clean     # Remove build output and old assets
#   make build     # Build the frontend
#   make deploy    # Copy build output to deploy directory
