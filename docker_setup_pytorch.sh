CONTNAME="$1"
IMGNAME="$2"
REPONAME="$3"
SCRIPTPATH=${4:-"/"}
JUPYTERPORT="9000"
TENSORBOARDPORT="6007"

docker run -it --name=$CONTNAME \
	-v "$SCRIPTPATH":/$REPONAME \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /etc/localtime:/etc/localtime:ro \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e XAUTHORITY \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--ipc=host \
	--gpus all \
	--net=host \
	-p $JUPYTERPORT:$JUPYTERPORT \
	-p $TENSORBOARDPORT:$TENSORBOARDPORT \
	--privileged=true \
	$IMGNAME /bin/bash
