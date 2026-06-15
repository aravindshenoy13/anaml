package main

import (
	"context"
	"net/http"
	"strings"
)

type contextKey string

const apiKeyContextKey contextKey = "apiKey"

type AuthMiddleware struct {
	validKeys map[string]bool
	skipPaths map[string]bool
}

func NewAuthMiddleware(keys []string, skipPaths []string) *AuthMiddleware {
	keyMap := make(map[string]bool)
	skipPathMap := make(map[string]bool)

	for _, key := range keys {
		keyMap[key] = true
	}

	for _, path := range skipPaths {
		skipPathMap[path] = true
	}

	return &AuthMiddleware{validKeys: keyMap, skipPaths: skipPathMap}
}

func (am *AuthMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if am.skipPaths[r.URL.Path] {
				next.ServeHTTP(w, r)
				return
			}
			authHeader := r.Header.Get("Authorization")
			parts := strings.SplitN(authHeader, " ", 2)
			if len(parts) != 2 || parts[0] != "Bearer" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				w.Write([]byte(`{"error": "malformed auth header"}`))
				return
			}
			if !am.validKeys[parts[1]] {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				w.Write([]byte(`{"error": "api key invalid"}`))
				return
			}
			ctx := context.WithValue(r.Context(), apiKeyContextKey, parts[1])
			r = r.WithContext(ctx)
			next.ServeHTTP(w, r)
		})
	}
}
