FROM freckie/rpi-gocv:1.15.0-buster

LABEL maintainer="freckie@frec.kr"

RUN mkdir -p $GOPATH/src/app
WORKDIR $GOPATH/src/app
COPY . .

RUN go build -o main main.go
ENTRYPOINT [ "./main", "--config=config.json" ]