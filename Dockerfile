FROM swift:5.5

WORKDIR /opt/app

COPY . .

RUN swift test
