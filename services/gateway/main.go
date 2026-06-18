package main

import (
	"log"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	upstreamRaw := os.Getenv("UPSTREAM_URL")
	redisURL := os.Getenv("REDIS_URL")
	rateString := os.Getenv("RATE_LIMIT_RPS")
	burstString := os.Getenv("RATE_LIMIT_BURST")
	keys := strings.Split(os.Getenv("API_KEYS"), ",")

	if upstreamRaw == "" {
		upstreamRaw = "http://localhost:8000"
	}
	upstream, err := url.Parse(upstreamRaw)
	if err != nil {
		log.Fatal(err)
	}

	if redisURL == "" {
		redisURL = "localhost:6379"
	}
	redisClient := redis.NewClient(&redis.Options{Addr: redisURL})

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
	auth := NewAuthMiddleware(keys, []string{"/health", "/readyz"})
	abRouter := NewABRouter(proxy)

	if abRouteRaw := os.Getenv("AB_ROUTE"); abRouteRaw != "" {
		path, backends := ParseABRoute(abRouteRaw)
		abRouter.AddRoute(path, backends)
	}

	handler := LoggingMiddleware(auth.Middleware()(limiter.Middleware()(abRouter)))

	http.Handle("/", handler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	addr := ":" + port

	slog.Info("gateway starting", "addr", addr, "upstream", upstreamRaw)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatal(err)
	}
}
