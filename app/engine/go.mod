module app/engine

go 1.14

require (
	app/config v0.0.0
	app/models v0.0.0
	github.com/aclements/go-gg v0.0.0-20170323211221-abd1f791f5ee
	gocv.io/x/gocv v0.25.0
)

replace (
	app/config v0.0.0 => ../config
	app/models v0.0.0 => ../models
)
