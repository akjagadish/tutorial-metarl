# tutorial on meta-rl

## pip install
`cd ~/tutorial-metarl/` <br/> 
`python -m venv pytorch-gpu` <br/> 
`source  pytorch-gpu/bin/activate` <br/> 
`pip3 install -r requirements.txt`  <br/> 
`pip3 install -e .`

## docker
### build docker image
`docker build -t tutorial_metarl .`
### create .env
add line `JUPYTER_PASSWORD=tutorial_metarl`
### run docker container
#### using GPU
`docker run --rm -d --name dev_user --env-file .env -v "/path/to/tutorial_metarl/:/notebooks/" -w "/notebooks/" -p 1236:8888  --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 tutorial_metarl:latest`
#### using CPU
`docker run --rm -d --name dev_user --env-file .env -v "/path/to/tutorial_metarl/:/notebooks/" -w "/notebooks/" -p 1236:8888  tutorial_metarl:latest`

Now, the notebooks should be accessble at the port 1236 (on your PC `http://localhost:1236/lab?` or on the server `http://your.server.ip.address:1236`).
Note: building docker container takes time and space as it includes a lot of other libraries often used in deep learning. But it can easily be reused as base image for your other containers. 

## conda install
`conda install PyTorch -c PyTorch`  <br/> 
`python3 setup.py install`