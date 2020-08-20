package config

import (
	"encoding/json"
	"io/ioutil"
)

// LoadConfig returns Config struct.
func LoadConfig(filePath string) (*Config, error) {
	cfg := &Config{}

	dataBytes, err := ioutil.ReadFile(filePath)
	if err != nil {
		return cfg, err
	}

	if err := json.Unmarshal(dataBytes, cfg); err != nil {
		return cfg, err
	}

	return cfg, nil
}
