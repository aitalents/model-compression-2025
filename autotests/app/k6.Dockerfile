FROM grafana/xk6:0.10.0 as xk6_builder
RUN xk6 build --output k6 --with github.com/szkiba/xk6-faker@v0.2.2 \
    --with github.com/grafana/xk6-output-prometheus-remote@v0.3.1


FROM grafana/k6:0.50.0

WORKDIR /app/
COPY src/main.js /app/
COPY --from=xk6_builder /xk6/k6 /usr/bin/k6

ENTRYPOINT k6 run -o xk6-prometheus-rw main.js
