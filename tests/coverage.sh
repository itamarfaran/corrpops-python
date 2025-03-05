#!/bin/bash

cd ../corrpops
coverage run --source="." -m pytest "../tests" "$@"
coverage report -m
