package engine

import (
	"fmt"
	"log"
	"strconv"

	"app/config"
	"app/models"

	"github.com/aclements/go-gg/generic/slice"
	"gocv.io/x/gocv"
)

type LabellingEngine struct {
	/* Private */
	model1Path         string
	model1InputLayer   string
	model1OutputLayers []string
	model2InputLayer   string
	model2OutputLayers []string
	model2Path         string
	pathForNoise       string
	pathForNum         string
	flagForSaveImg     bool
	readySignal        chan struct{}
	/* Public */
	Logger      *log.Logger
	Model1      *models.MLModel
	Model2      *models.MLModel
	ImageChan   chan *gocv.Mat
	ResultChan  chan string
	CloseSignal chan struct{}
}

// Init function initiates LabellingEngine with Config data.
func (le *LabellingEngine) Init(cfg config.Config, logger *log.Logger) error {
	le.Logger = logger

	// Load data from Config
	le.model1Path = cfg.LabellingEngine.Model1Path
	le.model1InputLayer = cfg.LabellingEngine.Model1InputLayer
	le.model1OutputLayers = cfg.LabellingEngine.Model1OutputLayers
	le.model2Path = cfg.LabellingEngine.Model2Path
	le.model2InputLayer = cfg.LabellingEngine.Model2InputLayer
	le.model2OutputLayers = cfg.LabellingEngine.Model2OutputLayers
	le.flagForSaveImg = cfg.LabellingEngine.FlagForSaveImg
	le.pathForNoise = cfg.LabellingEngine.PathForNoise
	le.pathForNum = cfg.LabellingEngine.PathForNum

	// Create channels
	le.ImageChan = make(chan *gocv.Mat, 100)
	le.ResultChan = make(chan string)
	le.CloseSignal = make(chan struct{}, 1)
	le.readySignal = make(chan struct{}, 1)

	// Load Models
	var err error
	le.Model1, err = models.NewMLModel(le.model1Path,
		le.model1InputLayer,
		le.model1OutputLayers)
	if err != nil {
		return err
	}
	le.Logger.Println("Model1 loaded.")
	le.Model2, err = models.NewMLModel(le.model2Path,
		le.model2InputLayer,
		le.model2OutputLayers)
	if err != nil {
		return err
	}
	le.Logger.Println("Model2 loaded.")

	le.Logger.Println("Initiated LabellingEngine successfully.")
	return nil
}

// Close terminates LabellingEngine.
func (le *LabellingEngine) Close() {
	le.CloseSignal <- struct{}{}

	// Close tensorflow sessions
	le.Model1.Close()
	le.Model2.Close()

	// Close channels
	close(le.ImageChan)
	close(le.ResultChan)
	close(le.readySignal)
	close(le.CloseSignal)
	le.Logger.Println("LabellingEngine closed.")
}

// WaitForReady waits for the labelling engine.
func (le *LabellingEngine) WaitForReady() {
	<-le.readySignal
}

// NewMatrix send a new 48*48*3 BGR image matrix to ImageChan.
func (le *LabellingEngine) NewMatrix(input *gocv.Mat) {
	le.ImageChan <- input
}

// Run is a function that starts main loop of labelling engine.
func (le *LabellingEngine) Run() {
	// Get outputs
	done := make(chan struct{}, 1)
	go func() {
	L1:
		for {
			select {
			case <-done:
				break L1

			case output := <-le.ResultChan:
				le.Logger.Printf("OUTPUT : %s\n", output)
			}
		}
	}()

	le.readySignal <- struct{}{}

	// Main Loop
L2:
	for {
		select {
		// Close signal.
		case <-le.CloseSignal:
			done <- struct{}{}
			break L2

		// New image matrix.
		case img := <-le.ImageChan:
			go func() {
				defer img.Close()

				err := le.makeResult(img)
				if err != nil {
					fmt.Println(err)
				}
			}()
		}
	}
}

func (le *LabellingEngine) makeResult(img *gocv.Mat) error {
	// Predict with checker model.
	output1, err := le.Model1.Predict(img, "gray")
	if err != nil {
		return err
	}

	// If the img doesn't contain number, return nil.
	if output1[0][0] >= output1[0][1] {
		return nil
	}

	// Predict with svhn model.
	_output2, err := le.Model2.Predict(img, "rgb")
	if err != nil {
		return err
	}
	output2 := make([]int, len(_output2))
	for idx := range _output2 {
		output2[idx] = slice.ArgMax(_output2[idx])
	}

	result, err := decodeLabel(output2)
	if err == nil {
		le.ResultChan <- result
	}

	return nil
}

func decodeLabel(value []int) (string, error) {
	valueChar := make([]string, len(value))
	for idx := range value {
		tempValue := value[idx]
		if value[idx] == 10 {
			tempValue = 0
		}
		valueChar[idx] = strconv.Itoa(tempValue)
	}

	if valueChar[0] == "0" {
		return "", fmt.Errorf("not a number image")
	}
	if valueChar[1] == "3" {
		return valueChar[2] + valueChar[3] + valueChar[4], nil
	} else if valueChar[1] == "4" {
		return valueChar[2] + valueChar[3] + valueChar[4] + "-" + valueChar[5], nil
	}
	return "", fmt.Errorf("not a number image")
}
