FROM ubuntu:20.04

COPY requirements.txt requirements.txt

RUN apt-get	update && \
	apt-get install -y --no-install-recommends vim wget unzip && \
	apt-get install -y --no-install-recommends \
		python3 python3-pip && \
	pip3 install --no-cache-dir -r requirements.txt && \
	apt-get remove -y python3-dev build-essential && \
	apt-get -y autoremove && \
	rm -rf /var/lib/apt/lists/*

# Setup NN
# Model need to download manual
COPY contribute/model.onnx /opt/model/model.onnx
# RUN mkdir /opt/model && \
#     wget "https://downloader.disk.yandex.ru/disk/a500ea6cc7dd3e5833bb8fcbcd232d398a7db275e289281e4be1ef4fe8fd7bb6/6102e661/HhWhdefDg_Ev82UezWhvV32k_-OxgytfEckDBH1JBZ3mKhXXqgUGYEL3GJv0t3kb9EvoWVqB83OQzK_vytRqUw%3D%3D?uid=0&filename=model_v0.onnx&disposition=attachment&hash=6Wy8qwRsgbsoE2XZ7Sk/p/lBAMxzgko5SYV7KMvir3vWrVCLJudu4cZOjytbLRk2q/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Foctet-stream&owner_uid=16862455&fsize=44714084&hid=29fdd0f5b3e0a83ec6d9dd4d5c95d956&media_type=data&tknv=v2" \
#         -O /opt/model/model.onnx

COPY source /opt/service

WORKDIR	/opt/service
ENTRYPOINT ["/opt/service/service.py"]
