#!/bin/bash
version=0.0.1

# Install dependence packages
# npm i

# Build server production
# ng build


# Start building docker image
docker build -t visual-information-retrieval-server:$version . --no-cache
docker tag visual-information-retrieval-server:$version it4u/visual-information-retrieval-server:$version
docker push it4u/visual-information-retrieval-server:$version
