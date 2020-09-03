FROM freckie/rpi-gocv:1.15.0-buster

LABEL maintainer="freckie@frec.kr"

RUN mkdir -p /app && mkdir -p /cfg
WORKDIR /app
COPY . .

# -a
RUN go build -o main main.go
ENTRYPOINT [ "./main", "--config=/cfg/config.json" ]
