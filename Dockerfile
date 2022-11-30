FROM redislabs/redisgears:edge
RUN apt-get install -y unzip
RUN wget -q https://redismodules.s3.amazonaws.com/rejson-oss/rejson-oss.Linux-ubuntu22.04-x86_64.2.4.2.zip -O tmp.zip; \
unzip -ju tmp.zip rejson.so -d ./target/release; rm tmp.zip
RUN wget -q https://redismodules.s3.amazonaws.com/redisearch/redisearch-light.Linux-ubuntu22.04-x86_64.2.6.3.zip -O tmp.zip; \
unzip -ju tmp.zip redisearch.so -d ./target/release; rm tmp.zip
EXPOSE 6379
CMD ["redis-server", "--protected-mode", "no",\
"--loadmodule", "./target/release/libredisgears.so", "./target/release/libredisgears_v8_plugin.so",\
"--loadmodule", "./target/release/rejson.so",\
"--loadmodule", "./target/release/redisearch.so"]