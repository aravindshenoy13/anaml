package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"

	"github.com/redis/go-redis/v9"
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

	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "localhost:6379"
	}
	redisClient := redis.NewClient(&redis.Options{Addr: redisURL})

	rateString := os.Getenv("RATE_LIMIT_RPS")
	burstString := os.Getenv("RATE_LIMIT_BURST")
	if rateString == "" {
		rateString = "10"
	}
	if burstString == "" {
		burstString = "100"
	}
	rate, _ := strconv.ParseFloat(rateString, 64)
	burst, _ := strconv.Atoi(burstString)

	proxy := httputil.NewSingleHostReverseProxy(upstream)

	limiter := NewRateLimiter(redisClient, rate, burst)
	handler := RateLimitMiddleware(limiter)(proxy)

	http.Handle("/", handler)

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
