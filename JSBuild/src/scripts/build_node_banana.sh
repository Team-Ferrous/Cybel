#!/bin/bash

git clone https://github.com/shrimbly/node-banana.git
cd node-banana
npm install

# $1 is the first argument passed to the script
if [ "$1" = "true" ]; then
  npm run dev
fi