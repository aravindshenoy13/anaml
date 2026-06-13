package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
)

func main() {
	upstreamRaw := os.Getenv("UPSTREAM_URL")
	if upstreamRaw == "" {
		upstreamRaw = "http://localhost:8000"
	}
	upstream, err := url.Parse(upstreamRaw)
	if err != nil {
		log.Fatal(err)
	}

	proxy := httputil.NewSingleHostReverseProxy(upstream)

	http.Handle("/", proxy)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	addr := ":" + port

	log.Printf("gateway listening on %s -> %s", addr, upstreamRaw)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}

}
