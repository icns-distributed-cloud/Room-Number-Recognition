package engine

import (
	"app/config"
	"image"

	"gocv.io/x/gocv"
)

// MainEngine is a wrapper for opencv control,
// 	and also controls LabellingEngine.
type MainEngine struct {
	/* Private */
	deviceNumber int
	paddingSize  int
	windowHSize  int
	windowWSize  int
	/* Public */
	CloseSignal chan struct{}
	LE          *LabellingEngine
}

// Init function initiates MainEngine and LabellingEngine with Config data.
func (me *MainEngine) Init(cfg config.Config) error {
	// Load data from Config
	me.deviceNumber = cfg.MainEngine.DeviceNumber
	me.paddingSize = cfg.MainEngine.PaddingSize
	me.windowHSize = cfg.MainEngine.WindowHorizontalSize
	me.windowWSize = cfg.MainEngine.WindowVertialSize

	// Create channels
	me.CloseSignal = make(chan struct{}, 1)

	// Init LabellingEngine
	me.LE = &LabellingEngine{}
	if err := me.LE.Init(cfg); err != nil {
		return err
	}

	return nil
}

// Close terminates MainEngine.
func (me *MainEngine) Close() {
	me.LE.Close()
	me.CloseSignal <- struct{}{}
	close(me.CloseSignal)
}

// Crop returns a matrix of cropped image from src.
func (me *MainEngine) Crop(src gocv.Mat, left, top, right, bottom int) gocv.Mat {
	croppedMat := src.Region(image.Rect(left, top, right, bottom))
	return croppedMat.Clone()
}

// DrawBbox

// Run
