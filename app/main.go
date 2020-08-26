package main

import (
	"flag"

	"app/config"
	"app/engine"
)

func main() {
	configPath := flag.String("config", "config.json", "Path for config file")
	flag.Parse()

	cfg, err := config.LoadConfig(*configPath)
	if err != nil {
		panic(err)
	}

	me := engine.MainEngine{}
	if err := me.Init(*cfg); err != nil {
		panic(err)
	}

	panic(me.Run())
}
