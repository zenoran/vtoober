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

start-llm:
	nohup uv run python -m llama_cpp.server \
		--model /home/nick/.cache/ask_llm/models/bartowski/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-GGUF/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf \
		--host 0.0.0.0 \
		--port 8000 \
		--n_gpu_layers -1 \
		--n_ctx 8192 \
		--chat_format chatml > llm-server.log 2>&1 &

# Usage:
#   make           # Clean, build, and deploy
#   make clean     # Remove build output and old assets
#   make build     # Build the frontend
#   make deploy    # Copy build output to deploy directory
