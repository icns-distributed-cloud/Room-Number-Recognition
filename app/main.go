package main

import (
	"flag"
	"log"
	"os"

	"app/config"
	"app/engine"
)

func main() {
	// Parse flag
	configPath := flag.String("config", "config.json", "Path for config file")
	flag.Parse()

	// Logger
	logger := log.New(os.Stdout, "LOG ", log.LstdFlags)

	// Load config
	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		panic(err)
	}
	logger.Println("Loaded configure file successfully.")
	logger.Println(*cfg)

	// Initialize engine.
	me := engine.MainEngine{}
	if err := me.Init(*cfg, logger); err != nil {
		panic(err)
	}

	panic(me.Run())
}
