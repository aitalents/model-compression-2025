### K6 AutoTests for Solution

#### Build and run Grafana and Prometheus

`docker-compose up`

#### Build test image
`docker build -t k6-tests -f k6.Dockerfile .`

#### Run tests
`docker run -ti -v $PWD/src:/app -e K6_PROMETHEUS_RW_SERVER_URL=http://0.0.0.0:9090/api/v1/write -e K6_PROMETHEUS_RW_TREND_AS_NATIVE_HISTOGRAM=true -e K6_OUT=xk6-prometheus-rw -e PARTICIPANT_NAME=baseline-solution-cpu -e api_host=http://0.0.0.0:8080/process k6-tests`

#### Check 0.0.0.0:3000 Grafana Dashboards
