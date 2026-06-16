package main

import (
	"hash/fnv"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
)

type WeightedBackend struct {
	Name   string
	URL    *url.URL
	Weight int
	Proxy  *httputil.ReverseProxy
}

type ABRoute struct {
	Path     string
	Backends []WeightedBackend
}

type ABRouter struct {
	routes       []ABRoute
	defaultProxy *httputil.ReverseProxy
}

func NewABRouter(defaultProxy *httputil.ReverseProxy) *ABRouter {
	return &ABRouter{defaultProxy: defaultProxy}
}

func (abr *ABRouter) AddRoute(path string, backends []WeightedBackend) {
	route := ABRoute{Path: path, Backends: backends}
	abr.routes = append(abr.routes, route)
}

func (abr *ABRouter) pickBackend(route ABRoute, key string) WeightedBackend {
	h := fnv.New32()
	h.Write([]byte(key))
	hashVal := h.Sum32()

	weightSum := 0
	for _, backend := range route.Backends {
		weightSum += backend.Weight
	}

	modVal := hashVal % uint32(weightSum)

	weightSum = 0
	for _, backend := range route.Backends {
		weightSum += backend.Weight
		if modVal < uint32(weightSum) {
			return backend
		}
	}
	//Fallback
	return route.Backends[0]
}

func (abr *ABRouter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := clientKey(r)
	for _, route := range abr.routes {
		if strings.HasPrefix(r.URL.Path, route.Path) {
			backend := abr.pickBackend(route, key)
			w.Header().Set("X-Model-Version", backend.Name)
			backend.Proxy.ServeHTTP(w, r)
			return
		}
	}
	abr.defaultProxy.ServeHTTP(w, r)
}

func ParseABRoute(raw string) (string, []WeightedBackend) {
	var backends []WeightedBackend
	parts := strings.Split(raw, ",")
	path := parts[0]
	for _, part := range parts[1:] {
		routeParts := strings.Split(part, "|")

		parsedURL, _ := url.Parse(routeParts[1])
		weight, _ := strconv.Atoi(routeParts[2])

		backend := WeightedBackend{
			Name:   routeParts[0],
			URL:    parsedURL,
			Weight: weight,
			Proxy:  httputil.NewSingleHostReverseProxy(parsedURL),
		}
		backends = append(backends, backend)
	}
	return path, backends
}
