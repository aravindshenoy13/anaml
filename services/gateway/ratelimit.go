package main

import (
	"context"
	"net"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
)

var luaScript = `
	local tokens = redis.call("HMGET", KEYS[1], "tokens", "last_refill")
	local rate = tonumber(ARGV[1])
	local burst = tonumber(ARGV[2])
	local now = tonumber(ARGV[3])

	local current_tokens = tonumber(tokens[1])
	local last_refill = tonumber(tokens[2])

	--Nil Bucket
	if current_tokens == nil then
		current_tokens = burst
		last_refill = now
	end

	--Refill Bucket
	local elapsed = now - last_refill
	local new_tokens = math.min(current_tokens + (elapsed * rate), burst)

	--Use 1 Token, Update Bucket
	if new_tokens>=1 then
		new_tokens = new_tokens-1
		redis.call("HMSET", KEYS[1], "tokens", new_tokens, "last_refill", now)
		redis.call("EXPIRE", KEYS[1], math.ceil(burst/rate))
		return 1
	else
		redis.call("HMSET", KEYS[1], "tokens", new_tokens, "last_refill", now)
		redis.call("EXPIRE", KEYS[1], math.ceil(burst/rate))
		return 0
	end
`

type RateLimiter struct {
	client *redis.Client
	rate   float64
	burst  int
}

func NewRateLimiter(client *redis.Client, rate float64, burst int) *RateLimiter {
	return &RateLimiter{client: client, rate: rate, burst: burst}
}

func (rl *RateLimiter) Allow(ctx context.Context, key string) (bool, error) {
	redisKey := "ratelimit:" + key
	now := float64(time.Now().UnixMilli()) / 1000.0

	res := rl.client.Eval(ctx, luaScript, []string{redisKey}, rl.rate, rl.burst, now)

	val, err := res.Int64()
	if err != nil {
		return true, nil
	}
	if val == 1 {
		return true, nil
	}
	return false, nil
}

func (limiter *RateLimiter) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			key := clientKey(r)
			val, _ := limiter.Allow(r.Context(), key)
			if !val {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusTooManyRequests)
				w.Write([]byte(`{"error": "rate limit exceeded"}`))
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func clientKey(r *http.Request) string {
	if key, ok := r.Context().Value(apiKeyContextKey).(string); ok && key != "" {
		return key
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
