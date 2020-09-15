package models

import (
	"app/config"
	"image"
	"sync"

	"gocv.io/x/gocv"
)

const imgWidth = 96
const imgHeight = 48

// YOLOModel is a wrapper for YOLO Net.
type YOLOModel struct {
	/* Private */
	mutex *sync.Mutex
	/* Public */
	Model            gocv.Net
	InputLayerName   string
	OutputLayerNames []string
	Threshold        float32
	NMSThreshold     float32
	NumClasses       int
}

// NewYOLOModel returns a new YOLOModel.
func NewYOLOModel(cfg config.ConfigModel2) (*YOLOModel, error) {
	model := &YOLOModel{}
	model.Model = gocv.ReadNet(cfg.WeightsPath, cfg.CfgPath)
	model.InputLayerName = cfg.InputLayer
	model.OutputLayerNames = getOutputNames(&model.Model)
	model.Threshold = cfg.Threshold
	model.NMSThreshold = cfg.NMSThreshold
	model.NumClasses = cfg.NumClasses
	model.mutex = &sync.Mutex{}
	return model, nil
}

// Close terminates YOLOModel.Model.
func (model *YOLOModel) Close() error {
	return model.Model.Close()
}

// Predict detects Boundary Boxes for each digit,
//	and returns DigitBoxes which contain coordinates, confidences, classIDs for each box.
func (model *YOLOModel) Predict(img *gocv.Mat) ([]DigitBox, error) {
	// Variables
	var rects []image.Rectangle
	var confidences []float32
	var classIDs []int
	var result []DigitBox

	// Pass img through the YOLO model
	outs, err := model.predict(img)
	if err != nil {
		return nil, err
	}

	// Get bboxes
	for i := range outs {
		cols := outs[i].Cols()
		rows := outs[i].Rows()

		for rowIdx := 0; rowIdx < rows; rowIdx++ {
			rowValues := make([]float32, cols)
			for colIdx := 0; colIdx < cols; colIdx++ {
				rowValues[colIdx] = outs[i].GetFloatAt(rowIdx, colIdx)
			}
			row := outs[i].RowRange(rowIdx, rowIdx+1)
			scores := row.ColRange(5, cols)

			_, confidence, _, classIDPoint := gocv.MinMaxLoc(scores)
			if confidence > model.Threshold {
				centerX := int(rowValues[0] * float32(imgHeight))
				centerY := int(rowValues[1] * float32(imgWidth))
				width := int(rowValues[2] * float32(imgHeight))
				height := int(rowValues[3] * float32(imgWidth))
				left := centerX - width/2.0
				top := centerY - height/2.0
				right := left + width
				bottom := top + height

				if !(left > 0 && height > 0 && top > 0 && width > 0) {
					continue
				}
				if left < 0 {
					left = 0
				}
				if top < 0 {
					top = 0
				}
				if right > imgHeight {
					right = imgHeight
				}
				if bottom > imgWidth {
					bottom = imgWidth
				}

				rects = append(rects, image.Rect(left, top, right, bottom))
				confidences = append(confidences, confidence)
				classIDs = append(classIDs, classIDPoint.X)
			}
		}
	}

	// Pass bboxes through the NMSBoxes algorithm
	indices := make([]int, len(rects))
	gocv.NMSBoxes(rects, confidences, model.Threshold, model.NMSThreshold, indices)
	for i := 0; i < len(indices); i++ {
		idx := indices[i]
		if idx == 0 {
			continue
		}

		result = append(result, DigitBox{
			BBox:       rects[idx],
			ClassID:    classIDs[idx],
			Confidence: confidences[idx],
		})
	}

	return result, nil
}

func (model *YOLOModel) predict(img *gocv.Mat) ([]gocv.Mat, error) {
	floatImg := img.Clone()
	img.ConvertTo(&floatImg, gocv.MatTypeCV32F)
	defer floatImg.Close()

	blob := gocv.BlobFromImage(*img, 1.0/255.0, image.Pt(96, 48), gocv.NewScalar(0, 0, 0, 0), true, false)
	defer blob.Close()

	model.mutex.Lock()
	model.Model.SetInput(blob, model.InputLayerName)

	outputs := model.Model.ForwardLayers(model.OutputLayerNames)
	model.mutex.Unlock()

	return outputs, nil
}

func getOutputNames(net *gocv.Net) []string {
	var names []string
	outLayers := net.GetUnconnectedOutLayers()
	layerNames := net.GetLayerNames()
	for i := 0; i < len(outLayers); i++ {
		idx := outLayers[i] - 1
		names = append(names, layerNames[idx])
	}
	return names
}
