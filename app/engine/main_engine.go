package engine

import (
	"app/config"
	"fmt"
	"image"
	"image/color"
	"log"
	"sync"
	"time"

	"gocv.io/x/gocv"
)

// MainEngine is a wrapper for opencv control,
// 	and also controls LabellingEngine.
type MainEngine struct {
	/* Private */
	deviceNumber int
	paddingSize  int
	windowHSize  int // height of window size (vertical)
	windowWSize  int // windth of window size (horizontal)
	/* Public */
	Logger      *log.Logger
	CloseSignal chan struct{}
	LE          *LabellingEngine
}

// Init function initiates MainEngine and LabellingEngine with Config data.
func (me *MainEngine) Init(cfg config.Config, logger *log.Logger) error {
	me.Logger = logger

	// Load data from Config
	me.deviceNumber = cfg.MainEngine.DeviceNumber
	me.paddingSize = cfg.MainEngine.PaddingSize
	me.windowWSize = cfg.MainEngine.WindowHorizontalSize
	me.windowHSize = cfg.MainEngine.WindowVertialSize

	// Create channels
	me.CloseSignal = make(chan struct{}, 1)

	// Init LabellingEngine
	me.LE = &LabellingEngine{}
	if err := me.LE.Init(cfg, logger); err != nil {
		return err
	}

	me.Logger.Println("Initiated MainEngine successfully.")

	return nil
}

// Close terminates MainEngine.
func (me *MainEngine) Close() {
	me.LE.Close()
	me.CloseSignal <- struct{}{}
	close(me.CloseSignal)
	me.Logger.Println("MainEngine closed.")
}

// DrawBbox analyzes controus in this frame,
//	detects number plate image, gets cropped image,
//	and send it to LabellingEngine.
// DrawBbox returns the time when the function is done.
func (me *MainEngine) DrawBbox(frame *gocv.Mat, prevTime time.Time) time.Time {
	// Keep the original frame image.
	originImg := frame.Clone()
	defer originImg.Close()

	// Get canny from grayscale image.
	grayImg := gocv.NewMat()
	canny := gocv.NewMat()
	gocv.CvtColor(originImg, &grayImg, gocv.ColorBGRToGray)
	gocv.Canny(grayImg, &canny, 50, 150)
	defer grayImg.Close()
	defer canny.Close()

	// Get all contour points
	contours := gocv.FindContours(canny, gocv.RetrievalList, gocv.ChainApproxNone)

	// Analyze each contours
	wg := sync.WaitGroup{}
	for idx := range contours {
		wg.Add(1)
		go func(_idx int) {
			defer wg.Done()

			// Get rectangle bounding contour.
			rect := gocv.BoundingRect(contours[_idx])
			_x, _y, _w, _h := rect.Min.X, rect.Min.Y, (rect.Max.X - rect.Min.X), (rect.Max.Y - rect.Min.Y)

			// Pass if the contour locates on boundary of image,
			//	or if the contour is too small/large.
			if me.filterNoise(_x, _y, _w, _h) {
				return
			}

			// Draw red rectangle box on frame image.
			croppedImg := crop(originImg, rect.Min.X, rect.Min.Y, rect.Max.X, rect.Max.Y)
			gocv.Rectangle(frame, rect, color.RGBA{255, 0, 0, 1}, 3)
			defer croppedImg.Close()

			// Reshape and send cropped BGR image to LabellingEngine
			resizedImg := gocv.NewMat()
			gocv.Resize(croppedImg, &resizedImg, image.Point{48, 48}, 0, 0, gocv.InterpolationLinear)
			me.LE.NewMatrix(&resizedImg)
		}(idx)
	}
	wg.Wait()

	// Calculate FPS and show FPS on frame
	currentTime := time.Now()
	elapsed := float64(time.Since(prevTime)) / float64(time.Second)
	fps := 1.0 / elapsed
	fpsStr := fmt.Sprintf("FPS : %.2f", fps)
	gocv.PutText(frame, fpsStr, image.Point{0, 40}, gocv.FontHersheyComplexSmall,
		0.8, color.RGBA{255, 0, 0, 1}, 1)

	return currentTime
}

// Run starts main loop of MainEngine.
func (me *MainEngine) Run() error {
	var err error

	// Open the video device
	cam, err := gocv.OpenVideoCapture(me.deviceNumber)
	if err != nil {
		return err
	}
	defer cam.Close()
	me.Logger.Println("Opened video device successfully.")

	// Open the display window
	window := gocv.NewWindow("Room-Number-Recog")
	window.ResizeWindow(me.windowWSize, me.windowHSize)
	defer window.Close()
	loading := gocv.IMRead("./loading.jpg", gocv.IMReadColor)
	window.IMShow(loading)
	window.WaitKey(3000)
	me.Logger.Println("Created new window.")

	// Run LabellingEngine.
	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		me.LE.Run()
		wg.Done()
	}()

	// Wait for ready
	me.Logger.Println("Waiting for LabellingEngine...")
	me.LE.WaitForReady()
	me.Logger.Println("LabellingEngine is ready now.")

	// Prepare variables for main loop
	prevTime := time.Now()
	frame := gocv.NewMat()
	defer frame.Close()

	// Main loop
	errCnt := 0
	me.Logger.Printf("Main loop started. camera device: %v\n", me.deviceNumber)
L:
	for {
		select {
		case <-me.CloseSignal:
			me.Logger.Println("Close signal for MainEngine detected.")
			break L

		default:
			// Read frame image
			// if ok := cam.Read(&frame); !ok {
			// 	me.Logger.Printf("Cannot read device %v\n", me.deviceNumber)
			// 	break L
			// }
			cam.Read(&frame)
			if frame.Empty() {
				errCnt++
				if errCnt%1000 == 0 {
					me.Logger.Printf("Error occured %dtimes.\n", errCnt)
				}
				continue
			}

			// Draw bbox
			prevTime = me.DrawBbox(&frame, prevTime)
			window.IMShow(frame)
			window.WaitKey(10)
		}
	}
	me.Logger.Println("Main loop has broken.")

	me.LE.Close()
	wg.Wait()
	return fmt.Errorf("terminated")
}

// filterNoise returns true if the contour is noise.
func (me MainEngine) filterNoise(x, y, w, h int) (isNoise bool) {
	winH, winW := me.windowHSize, me.windowWSize
	if w > 70 || h > 45 || w < 15 {
		return true
	}
	if float32(h/w) > 0.7 || float32(w/h) > 1.0 {
		return true
	}
	if h > 40 || w > 70 {
		return true
	}
	if y > 150 || x > 500 || x < 200 {
		return true
	}
	if float32(h) < float32(winH)*0.1 {
		return true
	}
	if float32(w) < float32(winW)*0.5 {
		return true
	}
	return false
}

// crop returns a matrix of cropped image from src.
func crop(src gocv.Mat, left, top, right, bottom int) gocv.Mat {
	croppedMat := src.Region(image.Rect(left, top, right, bottom))
	return croppedMat.Clone()
}
