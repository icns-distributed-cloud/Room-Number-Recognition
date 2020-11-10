module app/models

go 1.14

require (
	app/config v0.0.0
	github.com/tensorflow/tensorflow v1.15.3
	gocv.io/x/gocv v0.25.0
)

replace (
	app/config v0.0.0 => ../config
)
