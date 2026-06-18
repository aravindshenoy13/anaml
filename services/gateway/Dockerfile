FROM golang:1.26.1-alpine AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o gateway .

FROM scratch

COPY --from=builder /app/gateway .
EXPOSE 8080
USER 65534
ENTRYPOINT ["/gateway"]
