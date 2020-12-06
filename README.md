# tutorial on meta-rl

## pip install
pip3 install -r requiremnts
python setup.py install

## docker
### build docker image
docker build -t tutorial_metarl .
### create .env
`add JUPYTER_PASSWORD=tutorial_metarl`
### run docker container
#### using GPU
docker run --rm -d --name dev_user --env-file .env -v "/path/to/tutorial_metarl/:/notebooks/" -w "/notebooks/" -p 1236:8888  --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 tutorial_metarl:latest
#### using CPU
docker run --rm -d --name dev_user --env-file .env -v "/path/to/tutorial_metarl/:/notebooks/" -w "/notebooks/" -p 1236:8888  tutorial_metarl:latest

