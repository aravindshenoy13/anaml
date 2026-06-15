package main

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
