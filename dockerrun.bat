docker network create my-net
docker build -t app_g3_complet_mcp:latest -f Dockerfile.mcp .
docker run --name expert_immo_mcp -p 8001:8001 --network my-net --env-file .env app_g3_complet_mcp uv run python src/mcp_server/server.py 
docker run --name expert_immo_app -p 8000:8000 --network my-net --env-file .env app_g3_complet_mcp uv run python src/app/main.py 
docker run --name expert_immo_gradio -p 7860:7860 --network my-net --env-file .env app_g3_complet_mcp uv run python src/agentia/main.py 