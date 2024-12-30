# Requirements
- docker
- VS code
- devcontainer extention
- huggingface access token
- hadolint (optional)

# Install
1. Clone this repository
1. Move .devcontainer/base/
1. Build base image`dokcer compose --env-file .devcontainer/.env build`
1. Move .devcontainer/
1. Build image`docker compose build`
1. Launch container using devcontainer (Dependencies will be installed launch command in docker-compose.yaml)


※ Modify docker-compose.yaml to change docker image name if you need.
