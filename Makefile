SHELL := /bin/bash
export KERAS_BACKEND ?= torch

SRC=$(wildcard content/*)

all: html

html: ${SRC} _config.yml _toc.yml
	jupyter-book build .

pdf: ${SRC} _config.yml _toc.yml
	jupyter-book build . --builder pdflatex

clean:
	rm -fR _build/
