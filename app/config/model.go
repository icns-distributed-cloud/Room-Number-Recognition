package config

// Config for system
type Config struct {
	MainEngine      ConfigMainEngine      `json:"main_engine"`
	LabellingEngine ConfigLabellingEngine `json:"labelling_engine"`
}

// ConfigMainEngine is sub item of Config
type ConfigMainEngine struct {
	DeviceNumber         int `json:"device_number"`
	PaddingSize          int `json:"padding_size"`
	WindowHorizontalSize int `json:"window_horizontal_size"`
	WindowVertialSize    int `json:"window_vertical_size"`
}

// ConfigLabellingEngine is sub item of Config
type ConfigLabellingEngine struct {
	Model1                 ConfigModel1 `json:"model1"`
	Model2                 ConfigModel2 `json:"model2"`
	MaxOutputChannelLength int          `json:"max_output_channel_length"`
	FlagForSaveImg         bool         `json:"flag_for_save_img"`
	PathForNoise           string       `json:"path_for_noise"`
	PathForNum             string       `json:"path_for_num"`
}

// ConfigModel1 is sub item of ConfigLabellingEngine
type ConfigModel1 struct {
	Path         string   `json:"path"`
	InputLayer   string   `json:"input_layer"`
	OutputLayers []string `json:"output_layers"`
}

// ConfigModel2 is sub item of ConfigLabellingEngine
type ConfigModel2 struct {
	WeightsPath  string  `json:"weights_path"`
	CfgPath      string  `json:"cfg_path"`
	InputLayer   string  `json:"input_layer"`
	Threshold    float32 `json:"threshold"`
	NMSThreshold float32 `json:"nms_threshold"`
	NumClasses   int     `json:"num_classes"`
}
