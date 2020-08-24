module app

go 1.14

require (
	app/config v0.0.0
	app/engine v0.0.0
	app/models v0.0.0
	github.com/aclements/go-gg v0.0.0-20170323211221-abd1f791f5ee // indirect
	github.com/chewxy/math32 v1.0.6 // indirect
	github.com/gogo/protobuf v1.3.1 // indirect
	github.com/golang/protobuf v1.4.2 // indirect
	github.com/google/flatbuffers v1.12.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/tensorflow/tensorflow v1.15.3
	gocv.io/x/gocv v0.24.0
	gonum.org/v1/gonum v0.8.0 // indirect
	google.golang.org/protobuf v1.25.0 // indirect
)

replace (
	app/config v0.0.0 => ./config
	app/engine v0.0.0 => ./engine
	app/models v0.0.0 => ./models
)
