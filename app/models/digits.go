package models

import (
	"image"
)

// DigitBox contains ClassID, Confidence and Rect of BBOX.
type DigitBox struct {
	BBox       image.Rectangle
	ClassID    int
	Confidence float32
}
