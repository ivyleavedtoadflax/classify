init:
	pip3 install -r requirements

localinit:
	pip3 install --editable .

test:
	cd tests && python3 -m pytest

.PHONY: init test 
