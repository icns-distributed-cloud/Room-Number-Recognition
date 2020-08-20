package config

// Config for system
type Config struct {
	MainEngine      ConfigMainEngine      `json:"main_engine"`
	LabellingEngine ConfigLabellingEngine `json:"labelling_engine"`
}

type ConfigMainEngine struct {
	DeviceNumber         int `json:"device_number"`
	PaddingSize          int `json:"padding_size"`
	WindowHorizontalSize int `json:"window_horizontal_size"`
	WindowVertialSize    int `json:"window_vertical_size"`
}

type ConfigLabellingEngine struct {
	Model1Path     string `json:"model1_path"`
	Model2Path     string `json:"model2_path"`
	FlagForSaveImg bool   `json:"flag_for_save_img"`
	PathForNoise   string `json:"path_for_noise"`
	PathForNum     string `json:"path_for_num"`
}
