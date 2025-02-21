#!/bin/bash

cd ../corrpops
coverage run --source="." -m pytest "../tests" "$@"
coverage html
coverage report -m
